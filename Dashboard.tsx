"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  Copy,
  Download,
  Film,
  Loader2,
  RotateCcw,
  X,
} from "lucide-react";
import { Manrope } from "next/font/google";

const font = Manrope({ subsets: ["latin"], display: "swap" });

/* -------------------------------------------------------------------------- */
/* Types (mirror backend schemas.py)                                          */
/* -------------------------------------------------------------------------- */
type AspectRatio = "16:9" | "9:16" | "1:1";
type Resolution = "480p" | "720p";
type Fps = 8 | 12 | 16 | 24 | 30;
type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

interface GenerateRequestPayload {
  prompt: string;
  negative_prompt: string | null;
  aspect_ratio: AspectRatio;
  resolution: Resolution;
  fps: Fps;
  duration_seconds: number;
  num_inference_steps: number;
  guidance_scale: number;
  seed: number | null;
}

interface ResolvedParams {
  width: number;
  height: number;
  num_frames: number;
  fps: number;
  duration_seconds: number;
}

interface VideoInfo {
  url: string;
  width: number;
  height: number;
  num_frames: number;
  fps: number;
  duration_seconds: number;
  size_bytes: number;
  seed: number;
  generation_seconds: number;
}

interface GenerateResponse {
  job_id: string;
  status: JobStatus;
  queue_position: number;
  created_at: string;
  status_url: string;
  resolved: ResolvedParams;
}

interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  queue_position: number;
  progress: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  request: GenerateRequestPayload;
  resolved: ResolvedParams;
  video: VideoInfo | null;
  error: string | null;
}

interface HealthResponse {
  status: "ok" | "loading" | "error";
  model_loaded: boolean;
  device: string;
  queue_size: number;
  active_job_id: string | null;
  uptime_seconds: number;
}

interface FormState {
  prompt: string;
  negativePrompt: string;
  aspectRatio: AspectRatio;
  resolution: Resolution;
  fps: Fps;
  duration: number;
  steps: number;
  guidance: number;
}

interface Option<T> {
  value: T;
  label: string;
}

/* -------------------------------------------------------------------------- */
/* Constants                                                                  */
/* -------------------------------------------------------------------------- */
// Point at the FastAPI server, e.g. http://localhost:8000. Leave empty when a
// Next.js rewrite proxies /api/* to the backend.
const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");
// Optional. Anything in NEXT_PUBLIC_* is visible to every visitor.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

const POLL_MS = 1500;
const HEALTH_MS = 15000;
const MAX_POLL_FAILURES = 8;
const FRAME_CELLS = 32;
const STORAGE_KEY = "videogen:last-job";
const PROMPT_MAX = 2000;

// Must match TEMPORAL_STRIDE / SPATIAL_MULTIPLE on the server (used for the preview only).
const TEMPORAL_STRIDE = 4;
const SPATIAL_MULTIPLE = 16;

const ASPECT_RATIOS: Record<AspectRatio, [number, number]> = {
  "16:9": [16, 9],
  "9:16": [9, 16],
  "1:1": [1, 1],
};

const ASPECT_OPTIONS: Option<AspectRatio>[] = [
  { value: "16:9", label: "16:9 landscape" },
  { value: "9:16", label: "9:16 portrait" },
  { value: "1:1", label: "1:1 square" },
];
const RESOLUTION_OPTIONS: Option<Resolution>[] = [
  { value: "480p", label: "480p" },
  { value: "720p", label: "720p" },
];
const FPS_OPTIONS: Option<Fps>[] = ([8, 12, 16, 24, 30] as Fps[]).map((v) => ({
  value: v,
  label: `${v} fps`,
}));

const DEFAULT_FORM: FormState = {
  prompt: "",
  negativePrompt: "",
  aspectRatio: "16:9",
  resolution: "480p",
  fps: 24,
  duration: 4,
  steps: 50,
  guidance: 7.5,
};

const fieldClass =
  "w-full rounded-md border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-100 " +
  "placeholder:text-slate-500 [color-scheme:dark] focus:outline-none " +
  "focus-visible:border-amber-400 focus-visible:ring-2 focus-visible:ring-amber-400/60 " +
  "disabled:cursor-not-allowed disabled:opacity-60";

/* -------------------------------------------------------------------------- */
/* API helpers                                                                */
/* -------------------------------------------------------------------------- */
class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: { loc?: (string | number)[]; msg?: string }) => {
          const field = d.loc?.slice(1).join(".");
          return field ? `${field}: ${d.msg ?? "invalid value"}` : d.msg ?? "Invalid input";
        })
        .join(". ");
    }
  } catch {
    /* body was not JSON */
  }
  return `Request failed with status ${res.status}.`;
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body) headers.set("Content-Type", "application/json");
  if (API_KEY) headers.set("X-API-Key", API_KEY);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  } catch (e) {
    if (e instanceof DOMException && e.name === "AbortError") throw e;
    throw new ApiError("Can't reach the server. Check your connection and try again.", 0);
  }
  if (!res.ok) throw new ApiError(await readError(res), res.status);
  return (await res.json()) as T;
}

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : "Something went wrong. Try again.";
}

/* -------------------------------------------------------------------------- */
/* Pure helpers                                                               */
/* -------------------------------------------------------------------------- */
function estimateOutput(f: FormState): ResolvedParams {
  const [rw, rh] = ASPECT_RATIOS[f.aspectRatio];
  const short = f.resolution === "720p" ? 720 : 480;
  const [w, h] = rw >= rh ? [(short * rw) / rh, short] : [short, (short * rh) / rw];
  const snap = (v: number) =>
    Math.max(SPATIAL_MULTIPLE, Math.round(v / SPATIAL_MULTIPLE) * SPATIAL_MULTIPLE);
  const raw = Math.max(1, Math.round(f.duration * f.fps));
  const k = Math.max(1, Math.round((raw - 1) / TEMPORAL_STRIDE));
  const frames = k * TEMPORAL_STRIDE + 1;
  return {
    width: snap(w),
    height: snap(h),
    num_frames: frames,
    fps: f.fps,
    duration_seconds: Math.round((frames / f.fps) * 100) / 100,
  };
}

function formFromRequest(r: GenerateRequestPayload): FormState {
  return {
    prompt: r.prompt,
    negativePrompt: r.negative_prompt ?? "",
    aspectRatio: r.aspect_ratio,
    resolution: r.resolution,
    fps: r.fps,
    duration: r.duration_seconds,
    steps: r.num_inference_steps,
    guidance: r.guidance_scale,
  };
}

function formatElapsed(secs: number): string {
  if (secs < 60) return `${secs}s`;
  return `${Math.floor(secs / 60)}m ${String(secs % 60).padStart(2, "0")}s`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isActive(status: JobStatus | undefined): boolean {
  return status === "queued" || status === "running";
}

function useElapsedSeconds(active: boolean, resetKey: string): number {
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    if (!active) return;
    const start = Date.now();
    setSecs(0);
    const id = setInterval(() => setSecs(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [active, resetKey]);
  return secs;
}

/* -------------------------------------------------------------------------- */
/* Small presentational components                                            */
/* -------------------------------------------------------------------------- */
function SelectField<T extends string | number>(props: {
  id: string;
  label: string;
  value: T;
  options: Option<T>[];
  onChange: (v: T) => void;
}) {
  const { id, label, value, options, onChange } = props;
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm font-medium text-slate-300">
        {label}
      </label>
      <div className="relative">
        <select
          id={id}
          value={String(value)}
          onChange={(e) => {
            const match = options.find((o) => String(o.value) === e.target.value);
            if (match) onChange(match.value);
          }}
          className={`${fieldClass} appearance-none pr-9`}
        >
          {options.map((o) => (
            <option key={String(o.value)} value={String(o.value)}>
              {o.label}
            </option>
          ))}
        </select>
        <ChevronDown
          aria-hidden
          className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
        />
      </div>
    </div>
  );
}

function SliderField(props: {
  id: string;
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
}) {
  const { id, label, hint, value, min, max, step, display, onChange } = props;
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <label htmlFor={id} className="text-sm font-medium text-slate-300">
          {label}
        </label>
        <span className="text-sm font-semibold tabular-nums text-slate-100">{display}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-valuetext={display}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer accent-amber-400 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400/60"
      />
      <p className="mt-1 text-xs text-slate-400">{hint}</p>
    </div>
  );
}

function FilmStrip({ progress, running }: { progress: number; running: boolean }) {
  const filled = Math.floor(Math.min(Math.max(progress, 0), 1) * FRAME_CELLS);
  return (
    <div
      role="progressbar"
      aria-label="Generation progress"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(progress * 100)}
      className="flex gap-1"
    >
      {Array.from({ length: FRAME_CELLS }, (_, i) => {
        const tone =
          i < filled
            ? "bg-amber-400"
            : i === filled && running
              ? "bg-amber-400/40 motion-safe:animate-pulse"
              : "bg-slate-800";
        return (
          <span
            key={i}
            className={`h-5 flex-1 rounded-[2px] transition-colors duration-300 ${tone}`}
          />
        );
      })}
    </div>
  );
}

function ServerStatus({ health }: { health: HealthResponse | "unreachable" | null }) {
  let dot = "bg-slate-500";
  let text = "Checking server";
  if (health === "unreachable") {
    dot = "bg-rose-400";
    text = "Server unreachable";
  } else if (health) {
    if (health.status === "ok") {
      dot = "bg-emerald-400";
      text = health.queue_size > 0 ? `Model ready, ${health.queue_size} waiting` : "Model ready";
    } else if (health.status === "loading") {
      dot = "bg-amber-400 motion-safe:animate-pulse";
      text = "Loading model";
    } else {
      dot = "bg-rose-400";
      text = "Model failed to load";
    }
  }
  return (
    <p className="flex items-center gap-2 text-sm text-slate-300" role="status">
      <span aria-hidden className={`h-2 w-2 rounded-full ${dot}`} />
      {text}
    </p>
  );
}

function ResultVideo({ src }: { src: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="flex h-full w-full flex-col items-center justify-center gap-2 p-6 text-center">
        <AlertTriangle aria-hidden className="h-7 w-7 text-rose-400" />
        <p className="text-sm text-slate-300">
          The video file couldn&apos;t be loaded. It may have expired, so generate it again.
        </p>
      </div>
    );
  }
  return (
    <video
      src={src}
      controls
      autoPlay
      loop
      muted
      playsInline
      onError={() => setFailed(true)}
      className="h-full w-full bg-black object-contain"
    />
  );
}

/* -------------------------------------------------------------------------- */
/* Dashboard                                                                  */
/* -------------------------------------------------------------------------- */
export default function Dashboard() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobStatusResponse | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [cancelRequested, setCancelRequested] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reconnecting, setReconnecting] = useState(false);
  const [pollStopped, setPollStopped] = useState(false);
  const [pollNonce, setPollNonce] = useState(0);
  const [health, setHealth] = useState<HealthResponse | "unreachable" | null>(null);
  const [copied, setCopied] = useState(false);

  const stageRef = useRef<HTMLElement>(null);
  const restoringRef = useRef(false);

  const set = useCallback(
    <K extends keyof FormState>(key: K, value: FormState[K]) =>
      setForm((prev) => ({ ...prev, [key]: value })),
    [],
  );

  const trackJob = useCallback((id: string | null) => {
    setJobId(id);
    try {
      if (id) window.localStorage.setItem(STORAGE_KEY, id);
      else window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      /* storage unavailable (private mode, etc.) */
    }
  }, []);

  /* Resume the last job after a page reload. */
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        restoringRef.current = true;
        setJobId(stored);
      }
    } catch {
      /* ignore */
    }
  }, []);

  /* Server health. */
  useEffect(() => {
    const ctrl = new AbortController();
    const check = async () => {
      try {
        const data = await apiFetch<HealthResponse>("/api/health", { signal: ctrl.signal });
        setHealth(data);
      } catch (e) {
        if (!ctrl.signal.aborted) setHealth("unreachable");
      }
    };
    check();
    const id = setInterval(check, HEALTH_MS);
    return () => {
      ctrl.abort();
      clearInterval(id);
    };
  }, []);

  /* Poll the job until it reaches a terminal state. */
  useEffect(() => {
    if (!jobId) return;
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    setPollStopped(false);

    const tick = async () => {
      try {
        const data = await apiFetch<JobStatusResponse>(`/api/jobs/${jobId}`, {
          signal: ctrl.signal,
        });
        failures = 0;
        setReconnecting(false);
        if (restoringRef.current) {
          restoringRef.current = false;
          setForm(formFromRequest(data.request));
        }
        setJob(data);
        if (!isActive(data.status)) return;
        timer = setTimeout(tick, POLL_MS);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        if (e instanceof ApiError && e.status === 404) {
          restoringRef.current = false;
          setReconnecting(false);
          setJob(null);
          trackJob(null);
          setError("That job has expired or was removed. Start a new one.");
          return;
        }
        failures += 1;
        setReconnecting(true);
        if (failures >= MAX_POLL_FAILURES) {
          setPollStopped(true);
          return;
        }
        timer = setTimeout(tick, Math.min(POLL_MS * 2 ** failures, 15000));
      }
    };
    tick();

    return () => {
      ctrl.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId, pollNonce, trackJob]);

  const active = isActive(job?.status);
  const elapsed = useElapsedSeconds(active, `${job?.job_id ?? ""}:${job?.status ?? ""}`);
  const preview = estimateOutput(form);
  const stageDims = job?.resolved ?? preview;
  const stageRatio = stageDims.width / stageDims.height;
  const stageWidth = `min(100%, ${(stageRatio * 62).toFixed(2)}vh)`;

  const modelBlocked = health !== null && health !== "unreachable" && health.status !== "ok";
  const canSubmit =
    form.prompt.trim().length >= 3 && !submitting && !active && !modelBlocked && !(jobId && !job);

  const submit = useCallback(async () => {
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    setCancelRequested(false);

    const body: GenerateRequestPayload = {
      prompt: form.prompt.trim(),
      negative_prompt: form.negativePrompt.trim() || null,
      aspect_ratio: form.aspectRatio,
      resolution: form.resolution,
      fps: form.fps,
      duration_seconds: form.duration,
      num_inference_steps: form.steps,
      guidance_scale: form.guidance,
      seed: null,
    };

    try {
      const res = await apiFetch<GenerateResponse>("/api/generate", {
        method: "POST",
        body: JSON.stringify(body),
      });
      setJob({
        job_id: res.job_id,
        status: res.status,
        queue_position: res.queue_position,
        progress: 0,
        created_at: res.created_at,
        started_at: null,
        finished_at: null,
        request: body,
        resolved: res.resolved,
        video: null,
        error: null,
      });
      trackJob(res.job_id);
      if (window.matchMedia("(max-width: 1023px)").matches) {
        const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        stageRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
      }
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setSubmitting(false);
    }
  }, [canSubmit, form, trackJob]);

  const cancel = useCallback(async () => {
    if (!job) return;
    setCancelRequested(true);
    try {
      const data = await apiFetch<JobStatusResponse>(`/api/jobs/${job.job_id}`, {
        method: "DELETE",
      });
      setJob(data);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) return; // already finished; polling will catch up
      setCancelRequested(false);
      setError(messageOf(e));
    }
  }, [job]);

  const startOver = useCallback(() => {
    trackJob(null);
    setJob(null);
    setError(null);
    setCancelRequested(false);
    setReconnecting(false);
    setPollStopped(false);
  }, [trackJob]);

  const copySeed = useCallback(async (seed: number) => {
    try {
      await navigator.clipboard.writeText(String(seed));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }, []);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submit();
  };
  const onPromptKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void submit();
    }
  };

  /* ------------------------------ stage content ----------------------------- */
  const restoring = jobId !== null && job === null && !error;
  let stageBody: React.ReactNode;
  let stageTone = "border-slate-800";

  if (restoring) {
    stageBody = (
      <div className="flex flex-col items-center gap-3 text-slate-300">
        <Loader2 aria-hidden className="h-6 w-6 motion-safe:animate-spin" />
        <p className="text-sm">Loading your last video</p>
      </div>
    );
  } else if (!job) {
    stageTone = "border-dashed border-slate-700";
    stageBody = (
      <div className="flex max-w-xs flex-col items-center gap-3 px-6 text-center">
        <Film aria-hidden className="h-8 w-8 text-slate-500" />
        <p className="text-base font-medium text-slate-200">Your video will play here</p>
        <p className="text-sm text-slate-400">
          Describe a scene, adjust the settings, then select Generate video.
        </p>
      </div>
    );
  } else if (job.status === "queued") {
    stageBody = (
      <div className="px-6 text-center">
        <p className="text-4xl font-semibold tabular-nums text-slate-100 sm:text-5xl">
          {job.queue_position > 1 ? `#${job.queue_position}` : "Next"}
        </p>
        <p className="mt-2 text-sm text-slate-300">
          {job.queue_position > 1
            ? `${job.queue_position - 1} ${job.queue_position - 1 === 1 ? "job" : "jobs"} ahead of yours`
            : "Your video starts as soon as the GPU is free"}
        </p>
      </div>
    );
  } else if (job.status === "running") {
    stageBody = (
      <div className="px-6 text-center">
        <p className="text-5xl font-semibold tabular-nums text-slate-100 sm:text-6xl">
          {Math.round(job.progress * 100)}%
        </p>
        <p className="mt-2 text-sm text-slate-300">
          {cancelRequested ? "Stopping after the current step" : "Generating your video"}
        </p>
      </div>
    );
  } else if (job.status === "completed" && job.video) {
    stageTone = "border-slate-700";
    stageBody = <ResultVideo key={job.job_id} src={`${API_BASE}${job.video.url}`} />;
  } else {
    stageTone = "border-slate-700";
    const cancelled = job.status === "cancelled";
    stageBody = (
      <div className="flex max-w-sm flex-col items-center gap-3 px-6 text-center">
        <AlertTriangle
          aria-hidden
          className={`h-7 w-7 ${cancelled ? "text-slate-400" : "text-rose-400"}`}
        />
        <p className="text-base font-medium text-slate-100">
          {cancelled ? "Generation cancelled" : "Generation failed"}
        </p>
        <p className="text-sm text-slate-300">
          {job.error ?? "Something went wrong while generating the video."}
        </p>
        <div className="mt-1 flex gap-2">
          <button
            type="button"
            onClick={() => {
              startOver();
              setTimeout(() => void submit(), 0);
            }}
            disabled={modelBlocked || form.prompt.trim().length < 3}
            className="inline-flex items-center gap-1.5 rounded-md bg-amber-400 px-3 py-1.5 text-sm font-semibold text-slate-950 hover:bg-amber-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RotateCcw aria-hidden className="h-4 w-4" />
            Try again
          </button>
          <button
            type="button"
            onClick={startOver}
            className="rounded-md border border-slate-700 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
          >
            Dismiss
          </button>
        </div>
      </div>
    );
  }

  /* --------------------------------- render --------------------------------- */
  return (
    <div className={`${font.className} min-h-screen bg-slate-950 text-slate-100`}>
      <header className="border-b border-slate-800">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="flex items-center gap-2.5">
            <Film aria-hidden className="h-5 w-5 text-amber-400" />
            <h1 className="text-base font-semibold tracking-tight">Text to video</h1>
          </div>
          <ServerStatus health={health} />
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-10 px-4 py-6 sm:px-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)] lg:py-10">
        {/* ------------------------------ form ------------------------------ */}
        <form onSubmit={onSubmit} className="space-y-6" aria-label="Video settings">
          <div>
            <label htmlFor="prompt" className="mb-1.5 block text-sm font-medium text-slate-300">
              Prompt
            </label>
            <textarea
              id="prompt"
              rows={5}
              maxLength={PROMPT_MAX}
              value={form.prompt}
              onChange={(e) => set("prompt", e.target.value)}
              onKeyDown={onPromptKey}
              placeholder="A paper boat drifts down a rain-soaked street at dusk, shallow depth of field"
              className={`${fieldClass} resize-y leading-relaxed`}
            />
            <div className="mt-1 flex justify-between text-xs text-slate-400">
              <span>Press Ctrl+Enter to generate</span>
              <span className="tabular-nums">
                {form.prompt.length} / {PROMPT_MAX}
              </span>
            </div>
          </div>

          <div>
            <label htmlFor="negative" className="mb-1.5 block text-sm font-medium text-slate-300">
              Negative prompt
            </label>
            <input
              id="negative"
              type="text"
              maxLength={1000}
              value={form.negativePrompt}
              onChange={(e) => set("negativePrompt", e.target.value)}
              placeholder="blurry, low quality, watermark"
              className={fieldClass}
            />
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-3">
            <SelectField
              id="aspect"
              label="Aspect ratio"
              value={form.aspectRatio}
              options={ASPECT_OPTIONS}
              onChange={(v) => set("aspectRatio", v)}
            />
            <SelectField
              id="resolution"
              label="Resolution"
              value={form.resolution}
              options={RESOLUTION_OPTIONS}
              onChange={(v) => set("resolution", v)}
            />
            <SelectField
              id="fps"
              label="Frame rate"
              value={form.fps}
              options={FPS_OPTIONS}
              onChange={(v) => set("fps", v)}
            />
          </div>

          <div className="space-y-5 border-t border-slate-800 pt-6">
            <SliderField
              id="duration"
              label="Duration"
              hint="Longer clips take longer to render."
              value={form.duration}
              min={1}
              max={10}
              step={0.5}
              display={`${form.duration.toFixed(1)} s`}
              onChange={(v) => set("duration", v)}
            />
            <SliderField
              id="steps"
              label="Inference steps"
              hint="More steps add detail but slow generation."
              value={form.steps}
              min={10}
              max={100}
              step={1}
              display={String(form.steps)}
              onChange={(v) => set("steps", v)}
            />
            <SliderField
              id="guidance"
              label="Guidance scale"
              hint="Higher values follow your prompt more closely."
              value={form.guidance}
              min={1}
              max={20}
              step={0.5}
              display={form.guidance.toFixed(1)}
              onChange={(v) => set("guidance", v)}
            />
            <p className="text-sm text-slate-300">
              Output: <span className="tabular-nums">{preview.width} × {preview.height} px</span>,{" "}
              <span className="tabular-nums">{preview.num_frames} frames</span>,{" "}
              <span className="tabular-nums">{preview.duration_seconds} s</span>
            </p>
          </div>

          <div className="space-y-3">
            {error && (
              <div
                role="alert"
                className="flex items-start gap-2 rounded-md border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200"
              >
                <AlertTriangle aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                <p className="flex-1">{error}</p>
                <button
                  type="button"
                  onClick={() => setError(null)}
                  aria-label="Dismiss error"
                  className="rounded p-0.5 text-rose-300 hover:bg-rose-500/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-300"
                >
                  <X aria-hidden className="h-4 w-4" />
                </button>
              </div>
            )}
            {modelBlocked && (
              <p className="text-sm text-amber-300" role="status">
                {health !== "unreachable" && health?.status === "loading"
                  ? "The model is still loading. You can generate once it's ready."
                  : "The model isn't available. Ask the administrator to check the server."}
              </p>
            )}
            <button
              type="submit"
              disabled={!canSubmit}
              className="inline-flex w-full items-center justify-center gap-2 rounded-md bg-amber-400 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-amber-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-400"
            >
              {submitting ? (
                <>
                  <Loader2 aria-hidden className="h-4 w-4 motion-safe:animate-spin" />
                  Sending
                </>
              ) : active ? (
                "Generating"
              ) : (
                "Generate video"
              )}
            </button>
          </div>
        </form>

        {/* ------------------------------ stage ----------------------------- */}
        <section
          ref={stageRef}
          aria-label="Video preview"
          className="scroll-mt-4 lg:sticky lg:top-6 lg:self-start"
        >
          <div className="mx-auto" style={{ width: stageWidth }}>
            <div
              className={`relative flex items-center justify-center overflow-hidden rounded-lg border bg-slate-900 ${stageTone}`}
              style={{ aspectRatio: `${stageDims.width} / ${stageDims.height}` }}
            >
              {stageBody}
            </div>

            {job && active && (
              <div className="mt-4 space-y-3" aria-live="polite">
                <FilmStrip progress={job.progress} running={job.status === "running"} />
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm text-slate-300">
                    {job.status === "queued" ? "Waiting in queue" : "Rendering"},{" "}
                    <span className="tabular-nums">{formatElapsed(elapsed)}</span>
                  </p>
                  <button
                    type="button"
                    onClick={() => void cancel()}
                    disabled={cancelRequested}
                    className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    <X aria-hidden className="h-4 w-4" />
                    {cancelRequested ? "Cancelling" : "Cancel"}
                  </button>
                </div>
              </div>
            )}

            {(reconnecting || pollStopped) && jobId && (
              <div className="mt-3 flex items-center justify-between gap-3 text-sm text-amber-300" role="status">
                <p>
                  {pollStopped
                    ? "Lost contact with the server. Your job may still be running."
                    : "Reconnecting to the server"}
                </p>
                {pollStopped && (
                  <button
                    type="button"
                    onClick={() => setPollNonce((n) => n + 1)}
                    className="rounded-md border border-amber-400/50 px-3 py-1 font-medium hover:bg-amber-400/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                  >
                    Check again
                  </button>
                )}
              </div>
            )}

            {job?.status === "completed" && job.video && (
              <div className="mt-4 space-y-4">
                <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
                  <div>
                    <dt className="text-slate-400">Size</dt>
                    <dd className="font-medium tabular-nums">
                      {job.video.width} × {job.video.height}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Length</dt>
                    <dd className="font-medium tabular-nums">
                      {job.video.duration_seconds} s at {job.video.fps} fps
                    </dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">File</dt>
                    <dd className="font-medium tabular-nums">{formatBytes(job.video.size_bytes)}</dd>
                  </div>
                  <div>
                    <dt className="text-slate-400">Render time</dt>
                    <dd className="font-medium tabular-nums">
                      {formatElapsed(Math.round(job.video.generation_seconds))}
                    </dd>
                  </div>
                </dl>

                <div className="flex flex-wrap items-center gap-2">
                  <a
                    href={`${API_BASE}${job.video.url}`}
                    download={`video-${job.job_id.slice(0, 8)}.mp4`}
                    className="inline-flex items-center gap-1.5 rounded-md bg-amber-400 px-3 py-1.5 text-sm font-semibold text-slate-950 hover:bg-amber-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
                  >
                    <Download aria-hidden className="h-4 w-4" />
                    Download
                  </a>
                  <button
                    type="button"
                    onClick={() => void copySeed(job.video!.seed)}
                    className="inline-flex items-center gap-1.5 rounded-md border border-slate-700 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                  >
                    {copied ? (
                      <Check aria-hidden className="h-4 w-4 text-emerald-400" />
                    ) : (
                      <Copy aria-hidden className="h-4 w-4" />
                    )}
                    {copied ? "Copied" : `Copy seed ${job.video.seed}`}
                  </button>
                  <button
                    type="button"
                    onClick={startOver}
                    className="rounded-md border border-slate-700 px-3 py-1.5 text-sm font-medium text-slate-200 hover:bg-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
                  >
                    New video
                  </button>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
