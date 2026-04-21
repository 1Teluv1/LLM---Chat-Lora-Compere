"use client";

import { FormEvent, useEffect, useState } from "react";

import { CompareRequest, fetchArtifactOptions, type ArtifactOption } from "@/lib/api";

type Props = {
  loading: boolean;
  onSubmit: (payload: CompareRequest) => Promise<void>;
  advancedOnly?: boolean;
  onModelContextChange?: (context: {
    runtime: "llama_cpp" | "transformers";
    baseModelId: string;
    loraId: string;
  }) => void;
};

export function CompareForm({
  loading,
  onSubmit,
  advancedOnly = false,
  onModelContextChange
}: Props) {
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(42);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [runtime, setRuntime] = useState<"llama_cpp" | "transformers">("llama_cpp");
  const [baseModelId, setBaseModelId] = useState("");
  const [loraId, setLoraId] = useState("");
  const [baseOptions, setBaseOptions] = useState<ArtifactOption[]>([]);
  const [loraOptions, setLoraOptions] = useState<ArtifactOption[]>([]);

  useEffect(() => {
    if (advancedOnly) return;
    let cancelled = false;
    fetchArtifactOptions()
      .then((data) => {
        if (cancelled) return;
        setBaseOptions(data.base);
        setLoraOptions(data.lora);
      })
      .catch(() => {
        if (cancelled) return;
        setBaseOptions([]);
        setLoraOptions([]);
      });
    return () => {
      cancelled = true;
    };
  }, [advancedOnly]);

  useEffect(() => {
    if (advancedOnly || !onModelContextChange) return;
    onModelContextChange({ runtime, baseModelId, loraId });
  }, [advancedOnly, onModelContextChange, runtime, baseModelId, loraId]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (advancedOnly) return;
    await onSubmit({
      prompt,
      seed,
      top_k: topK,
      top_p: topP,
      temperature,
      max_tokens: maxTokens,
      runtime,
      base_model_id: baseModelId || null,
      lora_id: loraId || null,
      lora_strategy: "auto",
      device_hint: "auto"
    });
  }

  if (advancedOnly) {
    return (
      <div className="advanced-grid">
        <div className="range-card">
          <div className="range-top">
            <span>Repetition Penalty</span>
            <span className="range-value">1.10</span>
          </div>
          <input type="range" min="1" max="2" step="0.01" value="1.10" readOnly />
        </div>
        <div className="range-card">
          <div className="range-top">
            <span>Presence Penalty</span>
            <span className="range-value">0.20</span>
          </div>
          <input type="range" min="0" max="2" step="0.01" value="0.20" readOnly />
        </div>
        <div className="range-card">
          <div className="range-top">
            <span>Frequency Penalty</span>
            <span className="range-value">0.15</span>
          </div>
          <input type="range" min="0" max="2" step="0.01" value="0.15" readOnly />
        </div>
        <div className="range-card">
          <div className="range-top">
            <span>Stop Sequence</span>
            <span className="range-value">Optional</span>
          </div>
          <input type="text" placeholder="예: </s> 또는 END" />
        </div>
      </div>
    );
  }

  return (
    <form id="compare-form" onSubmit={handleSubmit} className="grid">
      <div className="field prompt">
        <label htmlFor="prompt">
          <span>Prompt</span>
          <span className="hint">가장 중요한 입력</span>
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="예: 다음 문장을 한국어로 전문적이고 자연스럽게 번역하고, JSON 형식으로만 출력하세요."
          rows={6}
          required
        />
      </div>

      <div className="field col-2">
        <label htmlFor="seed">
          <span>Seed</span>
          <span className="hint">재현성</span>
        </label>
        <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
      </div>

      <div className="field col-2">
        <label htmlFor="topk">
          <span>Top-k</span>
          <span className="hint">후보 수</span>
        </label>
        <input id="topk" type="number" value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
      </div>

      <div className="field col-2">
        <label htmlFor="topp">
          <span>Top-p</span>
          <span className="hint">누적 확률</span>
        </label>
        <input id="topp" type="number" step="0.01" value={topP} onChange={(e) => setTopP(Number(e.target.value))} />
      </div>

      <div className="field col-3">
        <label htmlFor="temperature">
          <span>Temperature</span>
          <span className="hint">다양성</span>
        </label>
        <input
          id="temperature"
          type="number"
          step="0.1"
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
        />
      </div>

      <div className="field col-3">
        <label htmlFor="maxTokens">
          <span>Max tokens</span>
          <span className="hint">최대 출력</span>
        </label>
        <input
          id="maxTokens"
          type="number"
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
        />
      </div>

      <div className="field col-2">
        <label htmlFor="runtime">
          <span>Runtime</span>
          <span className="hint">엔진</span>
        </label>
        <select
          id="runtime"
          value={runtime}
          onChange={(e) => setRuntime(e.target.value as "llama_cpp" | "transformers")}
        >
          <option value="llama_cpp">llama_cpp</option>
          <option value="transformers">transformers</option>
        </select>
      </div>

      <div className="field col-2">
        <label htmlFor="baseModelId">
          <span>Base Model ID</span>
          <span className="hint">artifacts 선택</span>
        </label>
        <select
          id="baseModelId"
          value={baseModelId}
          onChange={(e) => setBaseModelId(e.target.value)}
        >
          <option value="">자동 선택 (기본)</option>
          {baseOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="field col-2">
        <label htmlFor="loraId">
          <span>LoRA ID</span>
          <span className="hint">artifacts 선택</span>
        </label>
        <select
          id="loraId"
          value={loraId}
          onChange={(e) => setLoraId(e.target.value)}
        >
          <option value="">자동 선택 (기본)</option>
          {loraOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
    </form>
  );
}
