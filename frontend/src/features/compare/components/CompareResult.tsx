"use client";

import { CompareResponse } from "@/lib/api";

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
  const baseStatus = resolveBaseStatus(loading, phase, hasResult);
  const loraStatus = resolveLoraStatus(loading, phase, hasResult);
  const baseRenderedText = result?.base.text ?? baseText;
  const loraRenderedText = result?.lora.text ?? loraText;

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
          <pre>{baseRenderedText}</pre>
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
            {loraStatus}
          </div>
        </div>
        <div className="stream-box">
          <pre>{loraRenderedText}</pre>
        </div>
        <div className="footer-note">
          <span>Adapter: LoRA</span>
          <span>
            {result ? `${result.lora.duration_ms}ms · ${buildTokenSummary(loraRenderedText)}` : buildTokenSummary(loraRenderedText)}
          </span>
        </div>
      </article>
    </section>
  );
}
