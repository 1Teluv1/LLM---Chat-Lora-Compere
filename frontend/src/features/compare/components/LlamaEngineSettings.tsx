"use client";

import { useEffect, useRef, useState } from "react";

import { fetchLlamaDefaults, type LlamaLoadConfig } from "@/lib/api";

const LS_KEY = "lora_compare_llama_engine_v1";

type Props = {
  disabled: boolean;
  value: LlamaLoadConfig | null;
  onChange: (next: LlamaLoadConfig) => void;
};

function clamp(n: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, n));
}

export function LlamaEngineSettings({ disabled, value, onChange }: Props) {
  const [hydrated, setHydrated] = useState(false);
  /** n_ctx만 문자열로 편집(type=number 제어 이슈·재동기화 덮어쓰기 방지) */
  const [nCtxText, setNCtxText] = useState("");
  const nCtxEditingRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const def = await fetchLlamaDefaults();
        if (cancelled) return;
        let merged: LlamaLoadConfig = { ...def };
        try {
          const raw = localStorage.getItem(LS_KEY);
          if (raw) {
            const saved = JSON.parse(raw) as Partial<LlamaLoadConfig>;
            merged = {
              n_ctx:
                saved.n_ctx === null || saved.n_ctx === undefined
                  ? def.n_ctx
                  : typeof saved.n_ctx === "number"
                    ? saved.n_ctx
                    : def.n_ctx,
              n_threads: typeof saved.n_threads === "number" ? saved.n_threads : def.n_threads,
              n_gpu_layers: typeof saved.n_gpu_layers === "number" ? saved.n_gpu_layers : def.n_gpu_layers,
              n_batch:
                saved.n_batch === null
                  ? null
                  : typeof saved.n_batch === "number"
                    ? saved.n_batch
                    : def.n_batch,
              use_mmap: typeof saved.use_mmap === "boolean" ? saved.use_mmap : def.use_mmap,
              use_mlock: typeof saved.use_mlock === "boolean" ? saved.use_mlock : def.use_mlock
            };
          }
        } catch {
          /* ignore localStorage */
        }
        onChange(merged);
        setNCtxText(merged.n_ctx === null ? "" : String(merged.n_ctx));
      } catch {
        if (!cancelled) {
          const fallback: LlamaLoadConfig = {
            n_ctx: 8192,
            n_threads: 8,
            n_gpu_layers: 0,
            n_batch: null,
            use_mmap: true,
            use_mlock: false
          };
          onChange(fallback);
          setNCtxText(String(fallback.n_ctx));
        }
      } finally {
        if (!cancelled) setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
    // 초기 하이드레이션만 — onChange 넣으면 리렌더마다 폼이 리셋될 수 있음
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!value || !hydrated || nCtxEditingRef.current) return;
    setNCtxText(value.n_ctx === null ? "" : String(value.n_ctx));
  }, [value?.n_ctx, hydrated, value]);

  useEffect(() => {
    if (!hydrated || !value || disabled) return;
    localStorage.setItem(LS_KEY, JSON.stringify(value));
  }, [hydrated, value, disabled]);

  if (!value || !hydrated) {
    return (
      <p className="inference-log-hint inference-log-hint--tight">Llama.cpp 엔진 기본값을 불러오는 중…</p>
    );
  }

  function patch<K extends keyof LlamaLoadConfig>(key: K, v: LlamaLoadConfig[K]) {
    onChange({ ...value, [key]: v } as LlamaLoadConfig);
  }

  return (
    <div className={`llama-engine-settings ${disabled ? "llama-engine-settings--disabled" : ""}`}>
      <p className="inference-log-hint inference-log-hint--tight">
        이 패널은 <strong>Llama.cpp 전용</strong>입니다. 값은 GGUF 로드 시 적용되며, 바꾼 뒤 비교 실행 시{" "}
        <strong>모델을 다시 로드</strong>할 수 있습니다.
      </p>
      <div className="grid">
        <div className="field col-2">
          <label htmlFor="llamaNCtx">
            <span>n_ctx</span>
            <span className="hint">
              비우거나 0 → 서버 .env 기본. 지정 시 512~262144(입력 중에는 잘리지 않음, 포커스 나가면 보정)
            </span>
          </label>
          <input
            id="llamaNCtx"
            type="text"
            inputMode="numeric"
            autoComplete="off"
            disabled={disabled}
            className="llama-nctx-input"
            value={nCtxText}
            placeholder="예: 20000 (비움=env)"
            onFocus={() => {
              nCtxEditingRef.current = true;
            }}
            onChange={(e) => setNCtxText(e.target.value)}
            onBlur={() => {
              nCtxEditingRef.current = false;
              const t = nCtxText.trim();
              if (t === "") {
                patch("n_ctx", null);
                setNCtxText("");
                return;
              }
              const n = Number.parseInt(t, 10);
              if (!Number.isFinite(n)) {
                setNCtxText(value.n_ctx === null ? "" : String(value.n_ctx));
                return;
              }
              if (n === 0) {
                patch("n_ctx", 0);
                setNCtxText("0");
                return;
              }
              let out = n;
              if (n < 512) out = 512;
              if (n > 262144) out = 262144;
              patch("n_ctx", out);
              setNCtxText(String(out));
            }}
          />
        </div>
        <div className="field col-2">
          <label htmlFor="llamaNThreads">
            <span>n_threads</span>
            <span className="hint">CPU 스레드</span>
          </label>
          <input
            id="llamaNThreads"
            type="number"
            min={1}
            max={256}
            disabled={disabled}
            value={value.n_threads}
            onChange={(e) => {
              const n = Number.parseInt(e.target.value, 10);
              if (!Number.isFinite(n)) return;
              patch("n_threads", n);
            }}
            onBlur={() => {
              patch("n_threads", clamp(value.n_threads, 1, 256));
            }}
          />
        </div>
        <div className="field col-2">
          <label htmlFor="llamaNGpuLayers">
            <span>n_gpu_layers</span>
            <span className="hint">GPU 오프로드 레이어(-1=전부)</span>
          </label>
          <input
            id="llamaNGpuLayers"
            type="number"
            min={-1}
            max={65536}
            disabled={disabled}
            value={value.n_gpu_layers}
            onChange={(e) => {
              const n = Number.parseInt(e.target.value, 10);
              if (!Number.isFinite(n)) return;
              patch("n_gpu_layers", n);
            }}
            onBlur={() => {
              patch("n_gpu_layers", clamp(value.n_gpu_layers, -1, 65536));
            }}
          />
        </div>
        <div className="field col-2">
          <label htmlFor="llamaNBatch">
            <span>n_batch</span>
            <span className="hint">비우면 서버 .env(LLAMA_N_BATCH) 또는 미설정</span>
          </label>
          <input
            id="llamaNBatch"
            type="number"
            min={32}
            max={65536}
            disabled={disabled}
            value={value.n_batch ?? ""}
            placeholder="예: 512"
            onChange={(e) => {
              const t = e.target.value.trim();
              if (!t) {
                patch("n_batch", null);
                return;
              }
              const n = Number.parseInt(t, 10);
              if (!Number.isFinite(n)) return;
              patch("n_batch", n);
            }}
            onBlur={() => {
              if (value.n_batch === null) return;
              patch("n_batch", clamp(value.n_batch, 32, 65536));
            }}
          />
        </div>
        <div className="field col-2">
          <label htmlFor="llamaUseMmap">
            <span>use_mmap</span>
            <span className="hint">OS 페이지 캐시 공유(일반적으로 ON)</span>
          </label>
          <select
            id="llamaUseMmap"
            disabled={disabled}
            value={value.use_mmap ? "1" : "0"}
            onChange={(e) => patch("use_mmap", e.target.value === "1")}
          >
            <option value="1">ON</option>
            <option value="0">OFF</option>
          </select>
        </div>
        <div className="field col-2">
          <label htmlFor="llamaUseMlock">
            <span>use_mlock</span>
            <span className="hint">메모리 잠금(환경에 따라 실패 가능)</span>
          </label>
          <select
            id="llamaUseMlock"
            disabled={disabled}
            value={value.use_mlock ? "1" : "0"}
            onChange={(e) => patch("use_mlock", e.target.value === "1")}
          >
            <option value="0">OFF</option>
            <option value="1">ON</option>
          </select>
        </div>
      </div>
    </div>
  );
}
