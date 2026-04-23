export const BACKEND_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8001";

export type CompareRequest = {
  prompt: string;
  system_prompt?: string | null;
  enable_thinking?: boolean;
  seed: number;
  top_k: number;
  top_p: number;
  temperature: number;
  max_tokens: number;
  runtime?: "llama_cpp" | "transformers";
  base_model_id?: string | null;
  lora_id?: string | null;
  lora_strategy?: "auto" | "adapter" | "merged";
  device_hint?: "auto" | "cpu" | "cuda";
};

export type CompareResponse = {
  base: { text: string; duration_ms: number };
  lora: { text: string; duration_ms: number };
  params: CompareRequest;
  debug?: {
    runtime_effective?: "llama_cpp" | "transformers" | string;
    comparison_mode?: "lora_adapter" | "merged_gguf" | string;
    resolved_base_model_path?: string | null;
    resolved_compare_model_path?: string | null;
    lora_scale?: number | null;
    loaded_loras?: string[];
    ready?: boolean;
    load_failed_stage?: string | null;
  };
};

export type ArtifactOption = {
  value: string;
  label: string;
};

export type ArtifactOptionsResponse = {
  base: ArtifactOption[];
  lora: ArtifactOption[];
};

export type ArtifactDownloadRequest = {
  repo_id: string;
  target_type: "base" | "lora";
  filename?: string | null;
  allow_patterns?: string[] | null;
  output_subdir?: string | null;
  repo_type?: "model" | "dataset" | "space";
};

export type ArtifactDownloadResponse = {
  success: boolean;
  target_type: "base" | "lora";
  repo_id: string;
  resolved_path: string;
  detected_files: {
    base_gguf?: string | null;
    adapter_model_safetensors?: string | null;
    adapter_config_json?: string | null;
    adapter_model_gguf?: string | null;
  };
  warnings: string[];
};

function formatHttpError(status: number, body: string): string {
  const prefix = `HTTP ${status}`;
  if (!body.trim()) {
    return `${prefix}: 비교 요청에 실패했습니다.`;
  }
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    if (parsed.detail !== undefined) {
      if (typeof parsed.detail === "string") {
        return `${prefix}\n\n${parsed.detail}`;
      }
      return `${prefix}\n\n${JSON.stringify(parsed.detail, null, 2)}`;
    }
  } catch {
    /* 본문이 JSON이 아님 */
  }
  return `${prefix}\n\n${body}`;
}

export async function compareLora(
  payload: CompareRequest
): Promise<CompareResponse> {
  const response = await fetch(`${BACKEND_BASE_URL}/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(formatHttpError(response.status, errorText));
  }

  return (await response.json()) as CompareResponse;
}

export async function fetchArtifactOptions(): Promise<ArtifactOptionsResponse> {
  const response = await fetch("/api/artifacts-options", { cache: "no-store" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "artifacts 목록을 불러오지 못했습니다.");
  }
  return (await response.json()) as ArtifactOptionsResponse;
}

export async function downloadArtifact(
  payload: ArtifactDownloadRequest
): Promise<ArtifactDownloadResponse> {
  const response = await fetch(`${BACKEND_BASE_URL}/artifacts/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(formatHttpError(response.status, errorText));
  }
  return (await response.json()) as ArtifactDownloadResponse;
}

export type LoadingStage =
  | "idle"
  | "resolving"
  | "converting_peft"
  | "loading_base"
  | "base_loaded"
  | "loading_lora"
  | "ready"
  | "error";

export type LoadingStatus = {
  stage: LoadingStage;
  message: string;
  ready: boolean;
  runtime_name?: "llama_cpp" | "transformers" | string;
  device?: "cpu" | "cuda" | string;
  model_identifiers?: {
    base?: string | null;
    lora?: string | null;
  };
  model_loaded?: {
    base?: boolean;
    lora?: boolean;
    overall?: boolean;
  };
  capabilities?: {
    gpu_runtime_available?: boolean;
    gpu_metrics_available?: boolean;
  };
  comparison_mode?: string;
  elapsed_total_ms: number;
  elapsed_stage_ms: number;
  stage_durations_ms: Partial<Record<LoadingStage, number>>;
  base_file: { path: string | null; size_bytes: number | null };
  lora_file: { path: string | null; size_bytes: number | null };
  process?: {
    rss_bytes?: number;
    vms_bytes?: number;
    private_bytes?: number;
    num_threads?: number;
    cpu_percent?: number;
    system_ram_total_bytes?: number;
    system_ram_available_bytes?: number;
    system_ram_percent?: number;
    error?: string;
  };
  gpu?: {
    available?: boolean;
    name?: string | null;
    vram_total_bytes?: number | null;
    vram_used_bytes?: number | null;
    vram_free_bytes?: number | null;
    utilization_percent?: number | null;
    error?: string | null;
  };
  llama_cpp?: {
    version: string | null;
    supports_gpu_offload: boolean;
    n_gpu_layers_requested: number;
    n_gpu_layers_effective: number | null;
    gpu_forced_off: boolean;
  };
  error_reason: string | null;
};

export async function fetchRuntimeStatus(
  runtime: "llama_cpp" | "transformers"
): Promise<LoadingStatus> {
  const params = new URLSearchParams({ runtime });
  const response = await fetch(`${BACKEND_BASE_URL}/runtime/status?${params.toString()}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(formatHttpError(response.status, errorText));
  }
  return (await response.json()) as LoadingStatus;
}

/** SSE `data:` JSON 페이로드 (POST /compare/stream). */
export type StreamEvent =
  | {
      type: "meta";
      event: string;
      message?: string;
      stage?: LoadingStage;
      status?: LoadingStatus;
    }
  | {
      type: "phase";
      phase: "base" | "lora";
      event: "start" | "end";
      duration_ms?: number;
      text?: string;
    }
  | { type: "delta"; phase: "base" | "lora"; text: string }
  | {
      type: "done";
      base: { text: string; duration_ms: number };
      lora: { text: string; duration_ms: number };
      params: CompareRequest;
      debug?: CompareResponse["debug"];
    }
  | {
      type: "error";
      error: string;
      error_type?: string;
      traceback?: string;
      detail?: unknown;
    };

/**
 * Base → LoRA 순으로 토큰(디코딩) 단위 스트리밍.
 * 서버가 보내는 `error` 이벤트 후에는 예외를 던진다.
 */
export async function compareLoraStream(
  payload: CompareRequest,
  onEvent: (ev: StreamEvent) => void
): Promise<void> {
  const response = await fetch(`${BACKEND_BASE_URL}/compare/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(formatHttpError(response.status, errorText));
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("응답 스트림을 읽을 수 없습니다.");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      const raw = dataLine.slice(5).trim();
      if (!raw) continue;
      let ev: StreamEvent;
      try {
        ev = JSON.parse(raw) as StreamEvent;
      } catch {
        continue;
      }
      onEvent(ev);
      if (ev.type === "error") {
        const parts = [ev.error, ev.traceback].filter(Boolean);
        if (ev.detail !== undefined) {
          parts.push(
            typeof ev.detail === "string"
              ? ev.detail
              : JSON.stringify(ev.detail, null, 2)
          );
        }
        throw new Error(parts.join("\n\n"));
      }
    }
  }
}
