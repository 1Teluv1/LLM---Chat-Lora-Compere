"use client";

import { useState } from "react";

import { downloadArtifact } from "@/lib/api";

type Props = {
  open: boolean;
  onClose: () => void;
  onDownloaded: () => void;
};

export function ArtifactDownloadModal({ open, onClose, onDownloaded }: Props) {
  const [baseRepoId, setBaseRepoId] = useState("");
  const [baseFilename, setBaseFilename] = useState("");
  const [baseSubdir, setBaseSubdir] = useState("");
  const [loraRepoId, setLoraRepoId] = useState("");
  const [loraPatterns, setLoraPatterns] = useState("adapter_model.*,adapter_config.json");
  const [loraSubdir, setLoraSubdir] = useState("");
  const [loading, setLoading] = useState<"base" | "lora" | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  if (!open) return null;

  async function handleBaseDownload() {
    if (!baseRepoId.trim() || !baseFilename.trim()) {
      setError("Base 다운로드에는 repo_id와 filename이 필요합니다.");
      setMessage("");
      return;
    }
    try {
      setLoading("base");
      setError("");
      setMessage("");
      const response = await downloadArtifact({
        repo_id: baseRepoId.trim(),
        target_type: "base",
        filename: baseFilename.trim(),
        output_subdir: baseSubdir.trim() || null,
        repo_type: "model"
      });
      setMessage(`Base 다운로드 완료: ${response.resolved_path}`);
      if (response.warnings.length > 0) {
        setMessage((prev) => `${prev} (경고: ${response.warnings.join(" / ")})`);
      }
      onDownloaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Base 다운로드 중 오류가 발생했습니다.");
    } finally {
      setLoading(null);
    }
  }

  async function handleLoraDownload() {
    if (!loraRepoId.trim()) {
      setError("LoRA 다운로드에는 repo_id가 필요합니다.");
      setMessage("");
      return;
    }
    const allowPatterns = loraPatterns
      .split(",")
      .map((pattern) => pattern.trim())
      .filter(Boolean);
    try {
      setLoading("lora");
      setError("");
      setMessage("");
      const response = await downloadArtifact({
        repo_id: loraRepoId.trim(),
        target_type: "lora",
        allow_patterns: allowPatterns.length > 0 ? allowPatterns : null,
        output_subdir: loraSubdir.trim() || null,
        repo_type: "model"
      });
      setMessage(`LoRA 다운로드 완료: ${response.resolved_path}`);
      if (response.warnings.length > 0) {
        setMessage((prev) => `${prev} (경고: ${response.warnings.join(" / ")})`);
      }
      onDownloaded();
    } catch (e) {
      setError(e instanceof Error ? e.message : "LoRA 다운로드 중 오류가 발생했습니다.");
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="download-modal-overlay" role="dialog" aria-modal="true" aria-label="모델 다운로드">
      <div className="download-modal">
        <div className="download-modal-header">
          <h3>허깅 페이스 모델 다운로드</h3>
          <button type="button" className="btn btn-secondary" onClick={onClose} disabled={loading !== null}>
            닫기
          </button>
        </div>

        <div className="download-modal-grid">
          <div className="field">
            <label htmlFor="modalBaseRepoId">
              <span>HF Base Repo ID</span>
            </label>
            <input
              id="modalBaseRepoId"
              value={baseRepoId}
              onChange={(e) => setBaseRepoId(e.target.value)}
              placeholder="예: unsloth/Qwen3.6-35B-A3B-GGUF"
            />
          </div>
          <div className="field">
            <label htmlFor="modalBaseFilename">
              <span>HF Base Filename</span>
            </label>
            <input
              id="modalBaseFilename"
              value={baseFilename}
              onChange={(e) => setBaseFilename(e.target.value)}
              placeholder="예: Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
            />
          </div>
          <div className="field">
            <label htmlFor="modalBaseSubdir">
              <span>HF Base Output Subdir</span>
            </label>
            <input
              id="modalBaseSubdir"
              value={baseSubdir}
              onChange={(e) => setBaseSubdir(e.target.value)}
              placeholder="예: qwen3"
            />
          </div>
          <div className="field download-action">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleBaseDownload}
              disabled={loading !== null}
            >
              {loading === "base" ? "Base 다운로드 중..." : "HF Base 다운로드"}
            </button>
          </div>

          <div className="field">
            <label htmlFor="modalLoraRepoId">
              <span>HF LoRA Repo ID</span>
            </label>
            <input
              id="modalLoraRepoId"
              value={loraRepoId}
              onChange={(e) => setLoraRepoId(e.target.value)}
              placeholder="예: org/my-lora-repo"
            />
          </div>
          <div className="field">
            <label htmlFor="modalLoraPatterns">
              <span>HF LoRA Allow Patterns</span>
            </label>
            <input
              id="modalLoraPatterns"
              value={loraPatterns}
              onChange={(e) => setLoraPatterns(e.target.value)}
              placeholder="adapter_model.*,adapter_config.json"
            />
          </div>
          <div className="field">
            <label htmlFor="modalLoraSubdir">
              <span>HF LoRA Output Subdir</span>
            </label>
            <input
              id="modalLoraSubdir"
              value={loraSubdir}
              onChange={(e) => setLoraSubdir(e.target.value)}
              placeholder="예: my-lora"
            />
          </div>
          <div className="field download-action">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleLoraDownload}
              disabled={loading !== null}
            >
              {loading === "lora" ? "LoRA 다운로드 중..." : "HF LoRA 다운로드"}
            </button>
          </div>
        </div>

        {message ? <p className="download-message">{message}</p> : null}
        {error ? <p className="download-error">{error}</p> : null}
      </div>
    </div>
  );
}
