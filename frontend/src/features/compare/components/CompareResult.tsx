"use client";

import { CompareResponse } from "@/lib/api";

type Props = {
  baseText: string;
  loraText: string;
  result: CompareResponse | null;
  loading: boolean;
  phase: string | null;
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

export function CompareResult({ baseText, loraText, result, loading, phase }: Props) {
  const hasResult = !!result;
  const baseStatus = resolveBaseStatus(loading, phase, hasResult);
  const loraStatus = resolveLoraStatus(loading, phase, hasResult);

  return (
    <section className="results">
      <article className="output-card">
        <div className="output-head">
          <div className="output-title">Base Output</div>
          <div className="status">
            <span className="badge-dot" />
            {baseStatus}
          </div>
        </div>
        <div className="stream-box">
          <pre>{result?.base.text ?? baseText}</pre>
        </div>
        <div className="footer-note">
          <span>Model: Base</span>
          <span>{result ? `${result.base.duration_ms}ms` : "Latency · Tokens/sec · Total tokens"}</span>
        </div>
      </article>
      <article className="output-card">
        <div className="output-head">
          <div className="output-title">LoRA Output</div>
          <div className="status">
            <span className="badge-dot" />
            {loraStatus}
          </div>
        </div>
        <div className="stream-box">
          <pre>{result?.lora.text ?? loraText}</pre>
        </div>
        <div className="footer-note">
          <span>Adapter: LoRA</span>
          <span>{result ? `${result.lora.duration_ms}ms` : "Diff View · Logprob · Sampling trace"}</span>
        </div>
      </article>
    </section>
  );
}
