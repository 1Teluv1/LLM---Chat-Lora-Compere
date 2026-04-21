"use client";

import { useEffect, useState } from "react";

import { CompareForm } from "@/features/compare/components/CompareForm";
import { CompareProgress } from "@/features/compare/components/CompareProgress";
import { CompareResult } from "@/features/compare/components/CompareResult";
import { useCompareStream } from "@/features/compare/hooks/useCompareStream";

const STORAGE_KEY = "lora_compare_accordion_state_v1";

export default function HomePage() {
  const [modelContext, setModelContext] = useState<{
    runtime: "llama_cpp" | "transformers";
    baseModelId: string;
    loraId: string;
  }>({
    runtime: "llama_cpp",
    baseModelId: "",
    loraId: ""
  });
  const {
    loading,
    error,
    result,
    baseText,
    loraText,
    phase,
    loadingStatus,
    metricHistory,
    latencyHistory,
    submit
  } = useCompareStream(modelContext);
  const [openState, setOpenState] = useState<boolean[]>([true, false, false, true]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (!saved) return;
      const parsed = JSON.parse(saved) as boolean[];
      if (Array.isArray(parsed) && parsed.length === 4) {
        setOpenState(parsed.map(Boolean));
      }
    } catch {
      /* ignore restore errors */
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(openState));
  }, [openState]);

  function toggle(index: number) {
    setOpenState((prev) => prev.map((item, idx) => (idx === index ? !item : item)));
  }

  function expandAll() {
    setOpenState([true, true, true, true]);
  }

  function collapseAll() {
    setOpenState([false, false, false, false]);
  }

  const thinkFromSelection =
    modelContext.baseModelId.toLowerCase().includes("think") ||
    modelContext.loraId.toLowerCase().includes("think");
  const thinkFromResult =
    (result?.params.base_model_id ?? "").toLowerCase().includes("think") ||
    (result?.params.lora_id ?? "").toLowerCase().includes("think");
  const showThinkField = thinkFromSelection || thinkFromResult;

  return (
    <main className="shell">
      <section className="card">
        <div className="hero">
          <div className="badge-row">
            <span className="badge">
              <span className="badge-dot" />
              Base vs LoRA Compare
            </span>
            <span className="badge">Streaming</span>
            <span className="badge">Token-level Decode</span>
          </div>

          <h1>
            정량적 LoRA 성능 검증을 위한
            <br />
            <span className="headline-accent">Base vs LoRA 평가 대시보드</span>
          </h1>
          <p className="subtitle">
            동일한 프롬프트와 샘플링 조건에서 Base 모델과 LoRA 적용 모델의 출력을
            일관되게 비교하고, 로딩 상태·시스템 리소스·결과 지연시간을 한 화면에서
            검증할 수 있도록 설계했습니다.
          </p>
        </div>

        <div className="section">
          <div className="toolbar">
            <div className="toolbar-left">
              <h2>실험 컨트롤</h2>
              <p>실험 파라미터를 표준화해 재현 가능한 비교 시나리오를 빠르게 구성합니다.</p>
            </div>
            <div className="toolbar-actions">
              <button className="btn btn-secondary" type="button" onClick={expandAll}>
                모두 열기
              </button>
              <button className="btn btn-secondary" type="button" onClick={collapseAll}>
                모두 닫기
              </button>
            </div>
          </div>

          <div className="accordion">
            <div className={`accordion-item ${openState[0] ? "is-open" : ""}`}>
              <button
                className="accordion-trigger"
                type="button"
                aria-expanded={openState[0]}
                onClick={() => toggle(0)}
              >
                <div className="accordion-left">
                  <div className="accordion-icon-wrap">✍️</div>
                  <div>
                    <div className="accordion-title">기본 입력</div>
                    <div className="accordion-desc">가장 자주 쓰는 Prompt와 핵심 파라미터</div>
                  </div>
                </div>
                <div className="accordion-meta">
                  <span className="pill">필수</span>
                  <span className="chevron">⌄</span>
                </div>
              </button>
              {openState[0] ? (
                <div className="accordion-content open">
                  <div className="accordion-panel">
                    <CompareForm
                      loading={loading}
                      onSubmit={submit}
                      onModelContextChange={setModelContext}
                    />
                  </div>
                </div>
              ) : null}
            </div>

            <div className={`accordion-item ${openState[1] ? "is-open" : ""}`}>
              <button
                className="accordion-trigger"
                type="button"
                aria-expanded={openState[1]}
                onClick={() => toggle(1)}
              >
                <div className="accordion-left">
                  <div className="accordion-icon-wrap">⚙️</div>
                  <div>
                    <div className="accordion-title">고급 파라미터</div>
                    <div className="accordion-desc">평소에는 접어두고, 필요할 때만 펼쳐서 조정</div>
                  </div>
                </div>
                <div className="accordion-meta">
                  <span className="pill">선택</span>
                  <span className="chevron">⌄</span>
                </div>
              </button>
              {openState[1] ? (
                <div className="accordion-content open">
                  <div className="accordion-panel">
                    <CompareForm
                      loading={loading}
                      onSubmit={submit}
                      advancedOnly
                      onModelContextChange={setModelContext}
                    />
                  </div>
                </div>
              ) : null}
            </div>

            <div className={`accordion-item ${openState[2] ? "is-open" : ""}`}>
              <button
                className="accordion-trigger"
                type="button"
                aria-expanded={openState[2]}
                onClick={() => toggle(2)}
              >
                <div className="accordion-left">
                  <div className="accordion-icon-wrap">🚀</div>
                  <div>
                    <div className="accordion-title">실행 상태</div>
                    <div className="accordion-desc">
                      현재 모드, 실행 순서, 스트리밍 옵션 확인
                    </div>
                  </div>
                </div>
                <div className="accordion-meta">
                  <span className="pill">상태</span>
                  <span className="chevron">⌄</span>
                </div>
              </button>
              {openState[2] ? (
                <div className="accordion-content open">
                  <div className="accordion-panel">
                    <CompareProgress
                      phase={phase}
                      loadingStatus={loadingStatus}
                      loading={loading}
                      modelContext={modelContext}
                      metricHistory={metricHistory}
                      latencyHistory={latencyHistory}
                    />
                  </div>
                </div>
              ) : null}
            </div>

            <div className={`accordion-item ${openState[3] ? "is-open" : ""}`}>
              <button
                className="accordion-trigger"
                type="button"
                aria-expanded={openState[3]}
                onClick={() => toggle(3)}
              >
                <div className="accordion-left">
                  <div className="accordion-icon-wrap">🧪</div>
                  <div>
                    <div className="accordion-title">결과 비교</div>
                    <div className="accordion-desc">Base와 LoRA 결과를 나란히 비교</div>
                  </div>
                </div>
                <div className="accordion-meta">
                  <span className="pill">비교 결과</span>
                  <span className="chevron">⌄</span>
                </div>
              </button>
              {openState[3] ? (
                <div className="accordion-content open">
                  <div className="accordion-panel">
                    {error ? (
                      <section className="error-card">
                        <h3>오류</h3>
                        <pre>{error}</pre>
                      </section>
                    ) : null}
                    {showThinkField ? (
                      <section className="inference-inline" aria-label="Think 추론 필드">
                        <div className="inference-head">
                          <h3>Think 추론 필드</h3>
                          <span className="chip">Visible</span>
                        </div>
                        <p className="inference-desc">
                          Think 모델 결과의 추론 근거, 관찰 포인트, 검증 메모를 이곳에 기록하세요.
                        </p>
                        <textarea
                          className="inference-input"
                          placeholder="예: Think 출력이 단계적 근거를 더 명확히 제시함."
                          rows={4}
                        />
                      </section>
                    ) : null}
                    <CompareResult
                      baseText={baseText}
                      loraText={loraText}
                      result={result}
                      loading={loading}
                      phase={phase}
                    />
                  </div>
                </div>
              ) : null}
            </div>
          </div>

          <div className="bottom-run-action">
            <button className="btn btn-primary" type="submit" form="compare-form" disabled={loading}>
              {loading ? "실행 중..." : "Base vs LoRA 비교 실행"}
            </button>
          </div>
        </div>
      </section>
    </main>
  );
}
