"""
text_encoder.py — frozen text conditioning for the video DiT.

Backends
  t5     T5 / UMT5 encoder-only (e.g. google/t5-v1_1-xxl, 4096-d). Runs in bf16 (fp16 overflows on T5-XXL).
  llm    Any decoder-only LLM or VLM language tower. Uses hidden states from a chosen layer
         (default: penultimate) with optional chat-template prefix cropping.
  dummy  Deterministic hash embeddings, no downloads. For unit tests and smoke runs.

Contract shared by every encoder:
    emb, mask = encoder(["a prompt", ...])   # emb [B, L, D] with L fixed = max_length, mask [B, L] bool
    emb, mask = encoder.null_embedding(B)    # embedding of "" for classifier-free guidance
Padded positions of `emb` are zeroed, and the DiT masks them out in cross-attention.

Requirements for t5/llm: transformers, sentencepiece (T5), optionally ftfy.
"""
from __future__ import annotations

import html
import re
import zlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:  # optional, fixes mojibake in scraped captions
    import ftfy
except Exception:  # pragma: no cover
    ftfy = None

__all__ = [
    "TextEncoderConfig", "build_text_encoder", "BaseTextEncoder",
    "T5TextEncoder", "LLMTextEncoder", "DummyTextEncoder", "clean_prompt", "resolve_dtype",
]

_DTYPES = {
    "float32": torch.float32, "fp32": torch.float32,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def resolve_dtype(d: Union[str, torch.dtype]) -> torch.dtype:
    return d if isinstance(d, torch.dtype) else _DTYPES[d.lower()]


def clean_prompt(text: str) -> str:
    text = text if isinstance(text, str) else str(text)
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaseTextEncoder(nn.Module):
    embed_dim: int
    max_length: int

    def __init__(self):
        super().__init__()
        self.register_buffer("_probe", torch.zeros(1), persistent=False)  # tracks device through .to()
        self._null_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    @property
    def device(self) -> torch.device:
        return self._probe.device

    def train(self, mode: bool = True):  # always eval: no dropout in a frozen encoder
        return super().train(False)

    def _encode(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, prompts: Union[str, Sequence[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(prompts, str):
            prompts = [prompts]
        return self._encode([clean_prompt(p) for p in prompts])

    @torch.no_grad()
    def null_embedding(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = self._null_cache
        if cache is None or cache[0].device != self.device:
            cache = self._encode([""])
            self._null_cache = cache
        emb, mask = cache
        return emb.expand(batch_size, -1, -1).clone(), mask.expand(batch_size, -1).clone()


# --------------------------------------------------------------------------- #
# T5
# --------------------------------------------------------------------------- #
class T5TextEncoder(BaseTextEncoder):
    def __init__(self, name: str = "google/t5-v1_1-xxl", max_length: int = 256, dtype="bfloat16"):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = T5EncoderModel.from_pretrained(name, torch_dtype=resolve_dtype(dtype))
        self.model.eval().requires_grad_(False)
        self.embed_dim = int(self.model.config.d_model)

    def _encode(self, prompts: List[str]):
        tok = self.tokenizer(
            prompts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        ids, mask = tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
        hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, mask.bool()


# --------------------------------------------------------------------------- #
# LLM / multimodal-LLM language tower
# --------------------------------------------------------------------------- #
class LLMTextEncoder(BaseTextEncoder):
    """Hidden states of a decoder-only LM as text conditioning.

    prompt_template: e.g. "<|im_start|>system\\nDescribe the video.<|im_end|>\\n<|im_start|>user\\n{}<|im_end|>"
                     ("{}" is replaced by the prompt).
    crop_start:      number of leading template tokens to drop from the output (must be a constant count).
    model_cls:       transformers class name; use a text-only class for VLMs if AutoModel loads the full tower.
    """

    def __init__(
        self,
        name: str,
        max_length: int = 256,
        dtype="bfloat16",
        hidden_layer: int = -2,
        prompt_template: Optional[str] = None,
        crop_start: int = 0,
        model_cls: str = "AutoModel",
        trust_remote_code: bool = False,
    ):
        super().__init__()
        import transformers

        self.max_length, self.hidden_layer = max_length, hidden_layer
        self.prompt_template, self.crop_start = prompt_template, crop_start
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cls = getattr(transformers, model_cls)
        self.model = cls.from_pretrained(name, torch_dtype=resolve_dtype(dtype), trust_remote_code=trust_remote_code)
        self.model.eval().requires_grad_(False)
        conf = getattr(self.model.config, "text_config", self.model.config)
        self.embed_dim = int(conf.hidden_size)

    def _encode(self, prompts: List[str]):
        texts = [self.prompt_template.replace("{}", p) if self.prompt_template else p for p in prompts]
        tok = self.tokenizer(
            texts, max_length=self.max_length + self.crop_start, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        ids, mask = tok.input_ids.to(self.device), tok.attention_mask.to(self.device)
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hidden = out.hidden_states[self.hidden_layer][:, self.crop_start:]
        mask = mask[:, self.crop_start:]
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, mask.bool()


# --------------------------------------------------------------------------- #
# Dummy (tests / smoke runs)
# --------------------------------------------------------------------------- #
class DummyTextEncoder(BaseTextEncoder):
    """Per-word deterministic random vectors (crc32-seeded, so stable across processes)."""

    def __init__(self, embed_dim: int = 64, max_length: int = 16, dtype="float32"):
        super().__init__()
        self.embed_dim, self.max_length, self.dtype = embed_dim, max_length, resolve_dtype(dtype)

    def _word_vec(self, word: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(zlib.crc32(word.encode("utf-8")) & 0x7FFFFFFF)
        return torch.randn(self.embed_dim, generator=g)

    def _encode(self, prompts: List[str]):
        B, L = len(prompts), self.max_length
        emb = torch.zeros(B, L, self.embed_dim)
        mask = torch.zeros(B, L, dtype=torch.bool)
        for i, p in enumerate(prompts):
            words = p.lower().split()[: L - 1] + ["</s>"]
            for j, w in enumerate(words):
                emb[i, j] = self._word_vec(w)
                mask[i, j] = True
        return emb.to(self.device, self.dtype), mask.to(self.device)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
@dataclass
class TextEncoderConfig:
    kind: str = "t5"                      # "t5" | "llm" | "dummy"
    name: str = "google/t5-v1_1-xxl"
    max_length: int = 256
    dtype: str = "bfloat16"
    # llm only
    hidden_layer: int = -2
    prompt_template: Optional[str] = None
    crop_start: int = 0
    model_cls: str = "AutoModel"
    trust_remote_code: bool = False
    # dummy only
    embed_dim: int = 64


def build_text_encoder(cfg: TextEncoderConfig) -> BaseTextEncoder:
    if cfg.kind == "t5":
        return T5TextEncoder(cfg.name, cfg.max_length, cfg.dtype)
    if cfg.kind == "llm":
        return LLMTextEncoder(
            cfg.name, cfg.max_length, cfg.dtype, cfg.hidden_layer, cfg.prompt_template,
            cfg.crop_start, cfg.model_cls, cfg.trust_remote_code,
        )
    if cfg.kind == "dummy":
        return DummyTextEncoder(cfg.embed_dim, cfg.max_length, cfg.dtype)
    raise ValueError(f"unknown text encoder kind: {cfg.kind!r}")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest():
    assert clean_prompt("  a   cat &amp;  dog \n") == "a cat & dog"
    enc = build_text_encoder(TextEncoderConfig(kind="dummy", embed_dim=32, max_length=8))
    emb, mask = enc(["a red fox", "hello"])
    assert emb.shape == (2, 8, 32) and mask.shape == (2, 8) and mask.dtype == torch.bool
    assert mask[0].sum() == 4 and mask[1].sum() == 2                      # words + </s>
    assert torch.all(emb[~mask] == 0)                                     # padding zeroed
    emb2, _ = enc(["a red fox"])
    assert torch.equal(emb[0], emb2[0])                                   # deterministic
    n_emb, n_mask = enc.null_embedding(3)
    assert n_emb.shape == (3, 8, 32) and n_mask.sum(1).tolist() == [1, 1, 1]
    print("text_encoder self-test OK")


if __name__ == "__main__":
    _selftest()
