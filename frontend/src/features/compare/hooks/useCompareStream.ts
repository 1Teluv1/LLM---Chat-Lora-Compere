"use client";

import { useEffect, useRef, useState } from "react";

import {
  CompareRequest,
  CompareResponse,
  LoadingStatus,
  StreamEvent,
  compareLoraStream,
  fetchRuntimeStatus
} from "@/lib/api";

type ModelContext = {
  runtime: "llama_cpp" | "transformers";
  baseModelId: string;
  loraId: string;
};

type MetricPoint = {
  ts: number;
  label: string;
  cpuPercent: number | null;
  ramPercent: number | null;
  vramUsedGiB: number | null;
  vramTotalGiB: number | null;
};

type LatencyPoint = {
  name: "Base" | "LoRA";
  durationMs: number;
};

const WINDOW_MS = 30_000;

export function useCompareStream(modelContext: ModelContext) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [baseText, setBaseText] = useState("");
  const [loraText, setLoraText] = useState("");
  const [phase, setPhase] = useState<string | null>(null);
  const [loadingStatus, setLoadingStatus] = useState<LoadingStatus | null>(null);
  const [metricHistory, setMetricHistory] = useState<MetricPoint[]>([]);
  const [latencyHistory, setLatencyHistory] = useState<LatencyPoint[]>([]);
  const [requestedMaxTokens, setRequestedMaxTokens] = useState<number | null>(null);
  const pollTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!loadingStatus) return;
    const now = Date.now();
    const point: MetricPoint = {
      ts: now,
      label: new Date(now).toLocaleTimeString("ko-KR", {
        minute: "2-digit",
        second: "2-digit"
      }),
      cpuPercent: loadingStatus.process?.cpu_percent ?? null,
      ramPercent: loadingStatus.process?.system_ram_percent ?? null,
      vramUsedGiB:
        typeof loadingStatus.gpu?.vram_used_bytes === "number"
          ? loadingStatus.gpu.vram_used_bytes / 1024 / 1024 / 1024
          : null,
      vramTotalGiB:
        typeof loadingStatus.gpu?.vram_total_bytes === "number"
          ? loadingStatus.gpu.vram_total_bytes / 1024 / 1024 / 1024
          : null
    };
    setMetricHistory((prev) => [...prev, point].filter((item) => now - item.ts <= WINDOW_MS));
  }, [loadingStatus]);

  useEffect(() => {
    let disposed = false;
    async function pollOnce() {
      if (loading || document.visibilityState !== "visible") return;
      try {
        const status = await fetchRuntimeStatus(modelContext.runtime);
        if (!disposed) setLoadingStatus(status);
      } catch {
        // 상태 폴링 실패는 UI를 멈추지 않기 위해 무시
      }
    }
    pollOnce();
    pollTimerRef.current = window.setInterval(pollOnce, 2500);
    return () => {
      disposed = true;
      if (pollTimerRef.current !== null) {
        window.clearInterval(pollTimerRef.current);
      }
    };
  }, [modelContext.runtime, loading]);

  async function submit(payload: CompareRequest): Promise<void> {
    setLoading(true);
    setError(null);
    setResult(null);
    setBaseText("");
    setLoraText("");
    setPhase(null);
    setRequestedMaxTokens(payload.max_tokens ?? null);
    setLoadingStatus(null);
    setLatencyHistory([]);
    try {
      await compareLoraStream(payload, (ev: StreamEvent) => {
        if (ev.type === "meta") {
          if (ev.status) setLoadingStatus(ev.status);
          if (ev.message) setPhase(ev.message);
        } else if (ev.type === "phase") {
          if (ev.event === "start") {
            setPhase(ev.phase === "base" ? "base" : "lora");
          }
        } else if (ev.type === "delta") {
          if (ev.phase === "base") setBaseText((p) => p + ev.text);
          if (ev.phase === "lora") setLoraText((p) => p + ev.text);
        } else if (ev.type === "done") {
          setResult({
            base: ev.base,
            lora: ev.lora,
            params: ev.params,
            debug: ev.debug
          });
          setLatencyHistory([
            { name: "Base", durationMs: ev.base.duration_ms },
            { name: "LoRA", durationMs: ev.lora.duration_ms }
          ]);
          setPhase("완료");
        }
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "요청 실패");
    } finally {
      setLoading(false);
    }
  }

  return {
    loading,
    error,
    result,
    baseText,
    loraText,
    phase,
    loadingStatus,
    metricHistory,
    latencyHistory,
    requestedMaxTokens,
    submit
  };
}
