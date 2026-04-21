"use client";

import { CompareResponse } from "@/lib/api";

type Props = {
  baseText: string;
  loraText: string;
  result: CompareResponse | null;
};

export function CompareResult({ baseText, loraText, result }: Props) {
  return (
    <section className="results">
      <article className="output-card">
        <div className="output-head">
          <div className="output-title">Base Output</div>
          <div className="status">
            <span className="badge-dot" />
            Ready
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
            Streaming
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
