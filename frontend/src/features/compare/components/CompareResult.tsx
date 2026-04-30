"use client";

import { CompareResponse, type InferenceLog, type PromptTokenInfo, type RunMode } from "@/lib/api";

type Props = {
  baseText: string;
  loraText: string;
  result: CompareResponse | null;
  loading: boolean;
  phase: string | null;
  requestedMaxTokens: number | null;
};

function resolveBaseStatus(loading: boolean, phase: string | null, hasResult: boolean): string {
  if (loading && phase?.toLowerCase().includes("base")) return "Streaming";
  if (loading && phase?.toLowerCase().includes("lora")) return "Ready";
  if (hasResult) return "Ready";
  return "Idle";
}

function resolveLoraStatus(loading: boolean, phase: string | null, hasResult: boolean): string {
  if (loading && phase?.toLowerCase().includes("lora")) return "Streaming";
  if (hasResult) return "Ready";
  return "Idle";
}

export function CompareResult({
  baseText,
  loraText,
  result,
  loading,
  phase,
  requestedMaxTokens
}: Props) {
  function estimateTokens(text: string): number {
    if (!text.trim()) return 0;
    return Math.max(1, Math.round(text.length / 4));
  }

  function buildTokenSummary(text: string): string {
    const used = estimateTokens(text);
    const max = result?.params.max_tokens ?? requestedMaxTokens;
    if (!max || max <= 0) return `~${used} tok`;
    const ratio = Math.min(100, (used / max) * 100);
    return `~${used}/${max} tok (${ratio.toFixed(1)}%)`;
  }

  const hasResult = !!result;
  const runMode: RunMode = result?.params.run_mode ?? "both";
  const baseStatus = resolveBaseStatus(loading, phase, hasResult);
  const loraStatus = resolveLoraStatus(loading, phase, hasResult);
  const baseRenderedText = result?.base.text ?? baseText;
  const loraRenderedText = result?.lora.text ?? loraText;
  const inf: InferenceLog | null | undefined = result?.inference_log;

  function formatUtcIso(iso: string): { local: string; utc: string } {
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return { local: iso, utc: iso };
      return {
        local: d.toLocaleString("ko-KR", {
          dateStyle: "medium",
          timeStyle: "medium"
        }),
        utc: d.toUTCString()
      };
    } catch {
      return { local: iso, utc: iso };
    }
  }

  function formatThroughputCharsPerSec(durationMs: number, chars: number): string {
    if (durationMs <= 0 || chars <= 0) return "—";
    const perSec = chars / (durationMs / 1000);
    return `${perSec.toFixed(1)} 자/초`;
  }

  function formatAvgCharsPerChunk(chunks: number, chars: number): string {
    if (chunks <= 0) return "—";
    return `${(chars / chunks).toFixed(2)} 자/청크`;
  }

  function InferenceMetricGrid({
    items
  }: {
    items: Array<{ k: string; v: string }>;
  }) {
    return (
      <dl className="inference-metric-grid">
        {items.map(({ k, v }) => (
          <div className="inference-metric-row" key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    );
  }

  function promptTokenRows(p: PromptTokenInfo): Array<{ k: string; v: string }> {
    const rows: Array<{ k: string; v: string }> = [
      {
        k: "렌더된 프롬프트 문자 수",
        v: p.rendered_prompt_chars.toLocaleString("ko-KR")
      }
    ];
    if (p.rendered_prompt_tokens != null) {
      rows.push({
        k: "프롬프트 토큰 수",
        v: p.rendered_prompt_tokens.toLocaleString("ko-KR")
      });
    }
    if (p.tokenizer_backend) {
      rows.push({ k: "토크나이저 출처", v: p.tokenizer_backend });
    }
    if (p.tokenize_params) {
      rows.push({ k: "llama tokenize 인자", v: p.tokenize_params });
    }
    if (p.tokenizer_name_or_path) {
      rows.push({ k: "HF 토크나이저", v: String(p.tokenizer_name_or_path) });
    }
    if (p.add_special_tokens != null) {
      rows.push({ k: "add_special_tokens", v: p.add_special_tokens ? "true" : "false" });
    }
    if (p.rendered_prompt_tokens_lora_pipeline != null) {
      rows.push({
        k: "프롬프트 토큰 (LoRA 파이프)",
        v: p.rendered_prompt_tokens_lora_pipeline.toLocaleString("ko-KR")
      });
    }
    if (p.lora_tokenizer_name_or_path) {
      rows.push({ k: "LoRA HF 토크나이저", v: String(p.lora_tokenizer_name_or_path) });
    }
    if (p.error) {
      rows.push({ k: "토큰 계산 오류", v: p.error });
    }
    if (p.lora_tokenizer_error) {
      rows.push({ k: "LoRA 토크나이저 오류", v: p.lora_tokenizer_error });
    }
    return rows;
  }

  function phaseDetailRows(label: "Base" | "LoRA", row: InferenceLog["base"]) {
    const ttft =
      row.time_to_first_chunk_ms == null ? "—" : `${row.time_to_first_chunk_ms.toLocaleString("ko-KR")} ms`;
    return (
      <div className="inference-phase-block" key={label}>
        <h5 className="inference-phase-title">{label}</h5>
        <InferenceMetricGrid
          items={[
            { k: "생성 소요", v: `${row.duration_ms.toLocaleString("ko-KR")} ms` },
            { k: "TTFT (첫 청크)", v: ttft },
            { k: "출력 길이", v: `${row.output_chars.toLocaleString("ko-KR")} 자` },
            { k: "줄 수", v: `${row.output_lines.toLocaleString("ko-KR")} 줄` },
            { k: "스트림 청크 수", v: `${row.stream_chunks.toLocaleString("ko-KR")}회` },
            { k: "max_tokens (요청)", v: `${row.max_tokens_requested.toLocaleString("ko-KR")}` },
            { k: "처리량 (대략)", v: formatThroughputCharsPerSec(row.duration_ms, row.output_chars) },
            { k: "청크당 문자(평균)", v: formatAvgCharsPerChunk(row.stream_chunks, row.output_chars) }
          ]}
        />
      </div>
    );
  }

  return (
    <section className="results">
      <article className="output-card">
        <div className="output-head">
          <div className="output-title">Base Output</div>
          <div className="status">
            <span className="badge-dot" />
            {runMode === "lora_only" ? "실행 안 함" : baseStatus}
          </div>
        </div>
        <div className="stream-box">
          <pre>{runMode === "lora_only" ? "(이 실행에서는 Base 추론을 건너뜀)" : baseRenderedText}</pre>
        </div>
        <div className="footer-note">
          <span>Model: Base</span>
          <span>
            {result ? `${result.base.duration_ms}ms · ${buildTokenSummary(baseRenderedText)}` : buildTokenSummary(baseRenderedText)}
          </span>
        </div>
      </article>
      <article className="output-card">
        <div className="output-head">
          <div className="output-title">LoRA Output</div>
          <div className="status">
            <span className="badge-dot" />
            {runMode === "base_only" ? "실행 안 함" : loraStatus}
          </div>
        </div>
        <div className="stream-box">
          <pre>{runMode === "base_only" ? "(이 실행에서는 LoRA 경로 추론을 건너뜀)" : loraRenderedText}</pre>
        </div>
        <div className="footer-note">
          <span>Adapter: LoRA</span>
          <span>
            {result ? `${result.lora.duration_ms}ms · ${buildTokenSummary(loraRenderedText)}` : buildTokenSummary(loraRenderedText)}
          </span>
        </div>
      </article>

      {inf ? (
        <details className="inference-log-block" open>
          <summary>추론 상세 (응답 완료 시각·지연·용량)</summary>
          <p className="inference-log-hint">
            TTFT는 스트림에서 <strong>첫 텍스트 청크</strong>가 나올 때까지의 시간입니다. 청크 수는 SSE로 넘어온
            디코딩 조각 개수라서 토큰 수와 1:1은 아닙니다. Base와 LoRA의 청크 수가 비슷한 것은 같은
            max_tokens·정지 조건에서 비슷한 길이로 끝났기 때문일 수 있습니다.
          </p>
          <div className="inference-log-body">
            <h5 className="inference-phase-title">전체</h5>
            <InferenceMetricGrid
              items={(() => {
                const { local, utc } = formatUtcIso(inf.finished_at_utc);
                return [
                  { k: "완료 시각 (이 브라우저 로컬)", v: local },
                  { k: "완료 시각 (UTC 문자열)", v: utc },
                  { k: "런타임", v: inf.runtime ?? "—" },
                  { k: "실행 모드", v: inf.mode ?? "—" },
                  {
                    k: "총 소요",
                    v: `${inf.run_total_ms.toLocaleString("ko-KR")} ms (${(inf.run_total_ms / 1000).toFixed(1)}초)`
                  }
                ];
              })()}
            />
            {inf.prompt ? (
              <div className="inference-phase-block">
                <h5 className="inference-phase-title">프롬프트 (렌더 후 · 추론 직전)</h5>
                <p className="inference-log-hint inference-log-hint--tight">
                  <code>llama_cpp</code>는 가능하면 GGUF <code>chat_template</code>(Jinja) 렌더 문자열,
                  <code>transformers</code>는 <code>apply_chat_template(enable_thinking=…)</code> 기준입니다.{" "}
                  Think OFF일 때만 구형 System/User 문자열에 끄기 문구가 붙을 수 있습니다.
                </p>
                <InferenceMetricGrid items={promptTokenRows(inf.prompt)} />
                {inf.prompt.note ? <p className="inference-log-hint inference-log-hint--tight">{inf.prompt.note}</p> : null}
              </div>
            ) : null}
            {phaseDetailRows("Base", inf.base)}
            {phaseDetailRows("LoRA", inf.lora)}
          </div>
        </details>
      ) : null}
    </section>
  );
}
