"use client";

import { LoadingStatus } from "@/lib/api";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type Props = {
  phase: string | null;
  loadingStatus: LoadingStatus | null;
  loading: boolean;
  modelContext: {
    runtime: "llama_cpp" | "transformers";
    baseModelId: string;
    loraId: string;
  };
  metricHistory: Array<{
    ts: number;
    label: string;
    cpuPercent: number | null;
    ramPercent: number | null;
    vramUsedGiB: number | null;
    vramTotalGiB: number | null;
  }>;
  latencyHistory: Array<{
    name: "Base" | "LoRA";
    durationMs: number;
  }>;
};

const CHART_COLORS = {
  cpu: "#f5c84c",
  ram: "#8f949d",
  vramUsed: "#f5c84c",
  vramTotal: "#6b7280",
  latency: "#f5c84c",
  axis: "#9ca3af",
  grid: "#2f333a",
  tooltipBg: "#111317",
  tooltipBorder: "#f5c84c",
  tooltipText: "#f3f4f6"
} as const;

function toPercent(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${value.toFixed(1)}%`;
}

function toGiB(value?: number | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${(value / 1024 / 1024 / 1024).toFixed(2)} GiB`;
}

function formatOneDecimal(value: unknown): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(1);
}

export function CompareProgress({
  phase,
  loadingStatus,
  loading,
  modelContext,
  metricHistory,
  latencyHistory
}: Props) {
  const runtime = loadingStatus?.runtime_name ?? modelContext.runtime;
  const baseModel =
    (loadingStatus?.model_identifiers?.base ?? modelContext.baseModelId) || "자동 선택";
  const loraModel =
    (loadingStatus?.model_identifiers?.lora ?? modelContext.loraId) || "자동 선택";
  const gpu = loadingStatus?.gpu;
  const process = loadingStatus?.process;
  const modelLoaded = loadingStatus?.model_loaded;
  const showJson = !!loadingStatus;

  return (
    <div className="run-status">
      <div className="run-meta">
        <span className="chip">Mode: Serial Compare</span>
        <span className="chip">Order: Base → LoRA</span>
        <span className="chip">Stream: Enabled</span>
        <span className="chip">{loading ? "State: Running" : "State: Idle"}</span>
        <span className="chip">Phase: {phase ?? "대기"}</span>
      </div>

      <div className="run-status-grid">
        <section className="status-card">
          <h4>Runtime / Device</h4>
          <p>Runtime: {runtime}</p>
          <p>Device: {loadingStatus?.device ?? "-"}</p>
          <p>Comparison: {loadingStatus?.comparison_mode ?? "-"}</p>
        </section>

        <section className="status-card">
          <h4>Model 상태</h4>
          <p>Base: {baseModel}</p>
          <p>LoRA/Compare: {loraModel}</p>
          <p>
            Loaded: base={modelLoaded?.base ? "Y" : "N"} / lora={modelLoaded?.lora ? "Y" : "N"} /
            overall={modelLoaded?.overall ? "Y" : "N"}
          </p>
        </section>

        <section className="status-card">
          <h4>CPU / RAM</h4>
          <p>CPU: {toPercent(process?.cpu_percent)}</p>
          <p>Process RSS: {toGiB(process?.rss_bytes)}</p>
          <p>System RAM: {toPercent(process?.system_ram_percent)}</p>
          <p>RAM Available: {toGiB(process?.system_ram_available_bytes)}</p>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={metricHistory}>
                <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
                <XAxis dataKey="label" minTickGap={24} tick={{ fill: CHART_COLORS.axis }} />
                <YAxis domain={[0, 100]} tick={{ fill: CHART_COLORS.axis }} />
                <Tooltip
                  formatter={(value) => `${formatOneDecimal(value)}%`}
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: 10,
                    color: CHART_COLORS.tooltipText
                  }}
                  labelStyle={{ color: CHART_COLORS.tooltipText }}
                  itemStyle={{ color: CHART_COLORS.tooltipText }}
                />
                <Legend />
                <Line type="monotone" dataKey="cpuPercent" name="CPU %" stroke={CHART_COLORS.cpu} dot={false} />
                <Line type="monotone" dataKey="ramPercent" name="RAM %" stroke={CHART_COLORS.ram} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="status-card">
          <h4>GPU</h4>
          <p>Available: {gpu?.available ? "Y" : "N"}</p>
          <p>Name: {gpu?.name ?? "-"}</p>
          <p>VRAM Used: {toGiB(gpu?.vram_used_bytes)}</p>
          <p>VRAM Total: {toGiB(gpu?.vram_total_bytes)}</p>
          {gpu?.available ? (
            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={metricHistory}>
                  <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="label" minTickGap={24} tick={{ fill: CHART_COLORS.axis }} />
                  <YAxis tick={{ fill: CHART_COLORS.axis }} />
                  <Tooltip
                    formatter={(value) => `${formatOneDecimal(value)} GiB`}
                    contentStyle={{
                      backgroundColor: CHART_COLORS.tooltipBg,
                      border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                      borderRadius: 10,
                      color: CHART_COLORS.tooltipText
                    }}
                    labelStyle={{ color: CHART_COLORS.tooltipText }}
                    itemStyle={{ color: CHART_COLORS.tooltipText }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="vramUsedGiB"
                    name="VRAM Used GiB"
                    stroke={CHART_COLORS.vramUsed}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="vramTotalGiB"
                    name="VRAM Total GiB"
                    stroke={CHART_COLORS.vramTotal}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p>GPU 메트릭을 수집할 수 없는 환경입니다.</p>
          )}
        </section>

        <section className="status-card stage-card">
          <h4>Stage</h4>
          <div className="stage-info-grid">
            <p>
              <span className="stage-label">Stage</span>
              <span className="stage-value">{loadingStatus?.stage ?? "idle"}</span>
            </p>
            <p>
              <span className="stage-label">Ready</span>
              <span className="stage-value">{loadingStatus?.ready ? "Y" : "N"}</span>
            </p>
            <p>
              <span className="stage-label">Message</span>
              <span className="stage-value">{loadingStatus?.message ?? "-"}</span>
            </p>
            <p className="stage-item-wide">
              <span className="stage-label">Error</span>
              <span className="stage-value">{loadingStatus?.error_reason ?? "-"}</span>
            </p>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={latencyHistory}>
                <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fill: CHART_COLORS.axis }} />
                <YAxis tick={{ fill: CHART_COLORS.axis }} />
                <Tooltip
                  formatter={(value) => `${formatOneDecimal(value)} ms`}
                  contentStyle={{
                    backgroundColor: CHART_COLORS.tooltipBg,
                    border: `1px solid ${CHART_COLORS.tooltipBorder}`,
                    borderRadius: 10,
                    color: CHART_COLORS.tooltipText
                  }}
                  labelStyle={{ color: CHART_COLORS.tooltipText }}
                  itemStyle={{ color: CHART_COLORS.tooltipText }}
                />
                <Bar dataKey="durationMs" name="Duration (ms)" fill={CHART_COLORS.latency} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>

      {showJson ? (
        <details className="status-debug">
          <summary>Raw status JSON</summary>
          <pre className="status-json">{JSON.stringify(loadingStatus, null, 2)}</pre>
        </details>
      ) : null}
    </div>
  );
}
