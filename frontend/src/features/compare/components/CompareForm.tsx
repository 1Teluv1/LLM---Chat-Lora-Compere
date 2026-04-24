"use client";

import { FormEvent, useEffect, useState } from "react";

import {
  CompareRequest,
  fetchArtifactOptions,
  type ArtifactOption,
  type LlamaLoadConfig
} from "@/lib/api";

const SETTINGS_STORAGE_KEY = "lora_compare_form_settings_v1";

type Props = {
  loading: boolean;
  onSubmit: (payload: CompareRequest) => Promise<void>;
  advancedOnly?: boolean;
  artifactsRefreshKey?: number;
  llamaLoad?: LlamaLoadConfig | null;
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
  artifactsRefreshKey = 0,
  llamaLoad = null,
  onModelContextChange
}: Props) {
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [seedInput, setSeedInput] = useState("42");
  const [topKInput, setTopKInput] = useState("40");
  const [topPInput, setTopPInput] = useState("0.9");
  const [temperatureInput, setTemperatureInput] = useState("0.7");
  const [maxTokensInput, setMaxTokensInput] = useState("512");
  const [enableThinking, setEnableThinking] = useState(false);
  const [runtime, setRuntime] = useState<"llama_cpp" | "transformers">("llama_cpp");
  const [baseModelId, setBaseModelId] = useState("");
  const [loraId, setLoraId] = useState("");
  const [baseOptions, setBaseOptions] = useState<ArtifactOption[]>([]);
  const [loraOptions, setLoraOptions] = useState<ArtifactOption[]>([]);

  function parseIntOrDefault(value: string, fallback: number): number {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function parseFloatOrDefault(value: string, fallback: number): number {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  useEffect(() => {
    if (advancedOnly) return;
    try {
      const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw) as {
        prompt?: string;
        systemPrompt?: string;
        seed?: number;
        topK?: number;
        topP?: number;
        temperature?: number;
        maxTokens?: number;
        enableThinking?: boolean;
        runtime?: "llama_cpp" | "transformers";
        baseModelId?: string;
        loraId?: string;
      };
      if (typeof parsed.prompt === "string") setPrompt(parsed.prompt);
      if (typeof parsed.systemPrompt === "string") setSystemPrompt(parsed.systemPrompt);
      if (typeof parsed.seed === "number") setSeedInput(String(parsed.seed));
      if (typeof parsed.topK === "number") setTopKInput(String(parsed.topK));
      if (typeof parsed.topP === "number") setTopPInput(String(parsed.topP));
      if (typeof parsed.temperature === "number") setTemperatureInput(String(parsed.temperature));
      if (typeof parsed.maxTokens === "number") setMaxTokensInput(String(parsed.maxTokens));
      if (typeof parsed.enableThinking === "boolean") setEnableThinking(parsed.enableThinking);
      if (parsed.runtime === "llama_cpp" || parsed.runtime === "transformers") setRuntime(parsed.runtime);
      if (typeof parsed.baseModelId === "string") setBaseModelId(parsed.baseModelId);
      if (typeof parsed.loraId === "string") setLoraId(parsed.loraId);
    } catch {
      /* ignore localStorage restore error */
    }
  }, [advancedOnly]);

  useEffect(() => {
    if (advancedOnly) return;
    const payload = {
      prompt,
      systemPrompt,
      seed: parseIntOrDefault(seedInput, 42),
      topK: parseIntOrDefault(topKInput, 40),
      topP: parseFloatOrDefault(topPInput, 0.9),
      temperature: parseFloatOrDefault(temperatureInput, 0.7),
      maxTokens: parseIntOrDefault(maxTokensInput, 512),
      enableThinking,
      runtime,
      baseModelId,
      loraId
    };
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(payload));
  }, [
    advancedOnly,
    prompt,
    systemPrompt,
    seedInput,
    topKInput,
    topPInput,
    temperatureInput,
    maxTokensInput,
    enableThinking,
    runtime,
    baseModelId,
    loraId
  ]);

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
  }, [advancedOnly, artifactsRefreshKey]);

  useEffect(() => {
    if (advancedOnly || !onModelContextChange) return;
    onModelContextChange({ runtime, baseModelId, loraId });
  }, [advancedOnly, onModelContextChange, runtime, baseModelId, loraId]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (advancedOnly) return;
    await onSubmit({
      prompt,
      system_prompt: systemPrompt.trim() || null,
      enable_thinking: enableThinking,
      seed: parseIntOrDefault(seedInput, 42),
      top_k: parseIntOrDefault(topKInput, 40),
      top_p: parseFloatOrDefault(topPInput, 0.9),
      temperature: parseFloatOrDefault(temperatureInput, 0.7),
      max_tokens: parseIntOrDefault(maxTokensInput, 512),
      runtime,
      base_model_id: baseModelId || null,
      lora_id: loraId || null,
      lora_strategy: "auto",
      device_hint: "auto",
      llama_load: runtime === "llama_cpp" && llamaLoad ? llamaLoad : undefined
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
        <label htmlFor="systemPrompt">
          <span>System Prompt</span>
          <span className="hint">역할/제약 조건 (선택)</span>
        </label>
        <textarea
          id="systemPrompt"
          value={systemPrompt}
          onChange={(e) => setSystemPrompt(e.target.value)}
          placeholder="예: 당신은 한국어 기술 문서 전문 어시스턴트입니다. 간결하고 정확하게 답변하세요."
          rows={4}
        />
      </div>

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
        <input id="seed" type="number" value={seedInput} onChange={(e) => setSeedInput(e.target.value)} />
      </div>

      <div className="field col-2">
        <label htmlFor="topk">
          <span>Top-k</span>
          <span className="hint">후보 수</span>
        </label>
        <input id="topk" type="number" value={topKInput} onChange={(e) => setTopKInput(e.target.value)} />
      </div>

      <div className="field col-2">
        <label htmlFor="topp">
          <span>Top-p</span>
          <span className="hint">누적 확률</span>
        </label>
        <input id="topp" type="number" step="0.01" value={topPInput} onChange={(e) => setTopPInput(e.target.value)} />
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
          value={temperatureInput}
          onChange={(e) => setTemperatureInput(e.target.value)}
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
          value={maxTokensInput}
          onChange={(e) => setMaxTokensInput(e.target.value)}
        />
      </div>

      <div className="field col-2">
        <label htmlFor="enableThinking">
          <span>추론 표시(Think)</span>
          <span className="hint">끄면 최종 답변 위주</span>
        </label>
        <select
          id="enableThinking"
          value={enableThinking ? "on" : "off"}
          onChange={(e) => setEnableThinking(e.target.value === "on")}
        >
          <option value="off">OFF</option>
          <option value="on">ON</option>
        </select>
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
