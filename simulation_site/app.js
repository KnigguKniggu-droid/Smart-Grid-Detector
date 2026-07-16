import {
  assertDispatchShape,
  assertResultsShape,
  createSimClock,
  dispatchCacheMatches,
  nextPollDelay,
  sectionOf,
} from "./logic.mjs";

import {
  clarkeTransformBatch,
  renderPhasePortrait,
  computeThd,
  generateThreePhaseSignal,
} from "./dsp-engine.js";

import {
  forwardPass,
  setThresholds,
  analyzeWaveform,
} from "./autoencoder-sim.js";

import {
  generateCppCode,
  getPlatformInfo,
  downloadCppFile,
} from "./hardware-export.js";

const MAX_RESULTS_BYTES = 5 * 1024 * 1024;
const FETCH_TIMEOUT_MS = 10_000;
const RESULTS_URL = "simulation_results.json";
// Adaptive polling: conditional requests back off while the recorded artifact
// is unchanged and snap back to the base cadence when a new run lands.
const POLL_BASE_MS = 5_000;
const POLL_MAX_MS = 60_000;
const POLL_BACKOFF = 1.5;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const RUN_MANIFEST = [
  { url: "simulation_results.json", label: "Seed 42 (canonical)" },
  { url: "simulation_results_alt.json", label: "Seed 137 (comparison)" },
];

const state = {
  data: null,
  records: [],
  current: 0,
  scan: 0,
  playing: !reducedMotion.matches,
  speed: 1,
  lastFrame: performance.now(),
  loading: false,
  controller: null,
  pollDelay: POLL_BASE_MS,
  pollTimer: null,
  validators: { etag: null, lastModified: null },
  dispatches: null,
  dispatchRunId: null,
  dispatchResultsSha256: null,
  dispatchRequestId: 0,
  dispatchValidators: { etag: null, lastModified: null },
  dispatchError: null,
  resultsSha256: null,
  activeRunUrl: "simulation_results.json",
  comparisonData: null,
};

const DISPATCHES_URL = "grid_dispatches.json";

// Fixed-time-stepping simulation clock, imported from the DOM-free module that
// the Node suite exercises directly.
const simClock = createSimClock();
window.GridReplay = {
  get state() { return state; },
  get tick() { return simClock.tick; },
  get liveState() { return liveState; },
};

const byId = (id) => document.getElementById(id);

function finiteNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function percent(value) {
  return `${(finiteNumber(value) * 100).toFixed(2)}%`;
}

function fixed(value, digits = 6) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function evidencePercent(value) {
  return Number.isFinite(value) ? percent(value) : "invalid";
}

function evidenceRatio(value, limit) {
  if (!Number.isFinite(value)) return 2; // Non-finite evidence pegs the meter.
  return value / Math.max(limit, 1e-12);
}

async function readBodyWithCap(response, cap) {
  if (!response.body) {
    const fallback = await response.text();
    if (new Blob([fallback]).size > cap) {
      throw new Error("Results artifact exceeds the 5 MiB dashboard limit.");
    }
    return fallback;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    received += value.byteLength;
    if (received > cap) {
      await reader.cancel();
      throw new Error("Results artifact exceeds the 5 MiB dashboard limit.");
    }
    chunks.push(value);
  }
  const merged = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return new TextDecoder().decode(merged);
}

function setText(id, value) {
  const element = byId(id);
  if (element) element.textContent = String(value);
}

async function sha256Hex(text) {
  if (!window.crypto?.subtle) {
    throw new Error("This browser cannot verify artifact integrity.");
  }
  const digest = await window.crypto.subtle.digest(
    "SHA-256", new TextEncoder().encode(text),
  );
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")).join("");
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(() => loadResults({ announce: false }), state.pollDelay);
}

function pollStatusSuffix() {
  return `next check in ~${Math.round(state.pollDelay / 1000)}s`;
}

async function loadResults({ announce = true, url } = {}) {
  if (state.loading) return;
  const targetUrl = url || state.activeRunUrl || RESULTS_URL;
  if (document.hidden && targetUrl === state.activeRunUrl) {
    schedulePoll();
    return;
  }
  state.loading = true;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  const refreshButton = byId("refresh-button");
  refreshButton.disabled = true;
  if (announce) setText("refresh-status", "Loading recorded run\u2026");

  let timedOut = false;
  const timeoutHandle = setTimeout(() => {
    timedOut = true;
    state.controller.abort();
  }, FETCH_TIMEOUT_MS);
  try {
    const headers = {};
    if (targetUrl === state.activeRunUrl && state.validators.etag) {
      headers["If-None-Match"] = state.validators.etag;
    }
    if (targetUrl === state.activeRunUrl && state.validators.lastModified) {
      headers["If-Modified-Since"] = state.validators.lastModified;
    }
    const response = await fetch(targetUrl, {
      cache: "no-store",
      credentials: "same-origin",
      headers,
      signal: state.controller.signal,
    });
    if (response.status === 304 && state.data && targetUrl === state.activeRunUrl) {
      state.pollDelay = nextPollDelay(
        state.pollDelay, false, POLL_BASE_MS, POLL_MAX_MS, POLL_BACKOFF,
      );
      setText(
        "refresh-status",
        `Unchanged \u00b7 checked ${new Date().toLocaleTimeString()} \u00b7 ${pollStatusSuffix()}`,
      );
      return;
    }
    if (!response.ok) throw new Error(`Results request failed with HTTP ${response.status}.`);
    const declaredLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_RESULTS_BYTES) {
      throw new Error("Results artifact exceeds the 5 MiB dashboard limit.");
    }
    const text = await readBodyWithCap(response, MAX_RESULTS_BYTES);
    const data = JSON.parse(text);
    assertResultsShape(data);
    const resultsSha256 = await sha256Hex(text);

    if (targetUrl === state.activeRunUrl) {
      state.validators.etag = response.headers.get("etag");
      state.validators.lastModified = response.headers.get("last-modified");
    }

    const changedRun =
      data.run.id !== state.data?.run?.id || resultsSha256 !== state.resultsSha256;
    state.pollDelay = nextPollDelay(
      state.pollDelay, changedRun, POLL_BASE_MS, POLL_MAX_MS, POLL_BACKOFF,
    );
    state.data = data;
    state.resultsSha256 = resultsSha256;
    state.records = data.replay.records;

    // Set thresholds for autoencoder simulation
    setThresholds(data.training.calibration.threshold, data.config.thd_limit);

    if (changedRun || targetUrl !== state.activeRunUrl) {
      state.current = 0;
      state.scan = 0;
      simClock.tick = 0;
      simClock.droppedTicks = 0;
    }
    state.activeRunUrl = targetUrl;
    renderAll();
    byId("state-light").className = "state-light is-ready";
    setText("run-status", "Recorded run ready");
    setText(
      "refresh-status",
      `Data current \u00b7 ${new Date().toLocaleTimeString()} \u00b7 ${pollStatusSuffix()}`,
    );

    if (targetUrl === RESULTS_URL) {
      loadComparison();
    }
  } catch (error) {
    if (error.name !== "AbortError" || timedOut) {
      byId("state-light").className = "state-light is-error";
      setText("run-status", "Results unavailable");
      const message = timedOut
        ? `Results request timed out after ${FETCH_TIMEOUT_MS / 1000} seconds.`
        : error.message;
      setText(
        "refresh-status",
        `${message} Run: python smart_grid_detector.py --export-all`,
      );
      console.error(error);
    }
  } finally {
    clearTimeout(timeoutHandle);
    state.loading = false;
    refreshButton.disabled = false;
    if (targetUrl === state.activeRunUrl) schedulePoll();
  }
}

function renderAll() {
  const data = state.data;
  const metrics = data.evaluation.metrics;
  const calibration = data.training.calibration;

  setText("run-id", data.run.id);
  setText("reconstruction-threshold", fixed(calibration.threshold, 8));
  setText("thd-threshold", percent(data.config.thd_limit));
  setText("metric-accuracy", percent(metrics.accuracy));
  setText("metric-precision", percent(metrics.precision));
  setText("metric-recall", percent(metrics.recall));
  setText("metric-f1", percent(metrics.f1_score));
  setText("metric-tp", metrics.true_positives);
  setText("metric-tn", metrics.true_negatives);
  setText("metric-fp", metrics.false_positives);
  setText("metric-fn", metrics.false_negatives);
  setText("metric-latency", `${fixed(metrics.inference_latency_ms, 2)} ms`);
  setText("metric-throughput", fixed(metrics.throughput_waveforms_per_second, 1));
  setText("calibration-mean", fixed(calibration.mean_error, 8));
  setText("calibration-std", fixed(calibration.std_error, 8));
  const training = data.training.epochs;
  setText("final-loss", training.length ? fixed(training.at(-1).mse, 8) : "n/a");

  const completed = new Date(data.run.completed_at_utc);
  setText("completed-at", Number.isNaN(completed.getTime()) ? "n/a" : completed.toLocaleString());
  setText("run-seed", data.config.seed);
  setText("run-device", data.model.device);
  setText("run-model", `${data.model.name} · ${data.model.parameters.toLocaleString()} params`);
  setText("source-hash", data.provenance.source_sha256.slice(0, 16));
  setText(
    "runtime-version",
    `Python ${data.provenance.python_version} · torch ${data.provenance.torch_version}`,
  );

  const slider = byId("record-slider");
  slider.max = String(Math.max(state.records.length - 1, 0));
  slider.value = String(state.current);
  renderCurrentRecord();
  renderTrainingCurve();
  renderAlerts();
  renderResilience(data);
  updateEnergyBalance();
  renderGridHealth(data);
  const downloadBtn = byId("download-report");
  if (downloadBtn) downloadBtn.disabled = false;
  if (window.GridTopology) window.GridTopology.update(state.data);
}

// --- Resilience & robustness panels (strict DOM, textContent only) ---------

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function buildTable(headers, rows) {
  const table = el("table", null, "resilience-table");
  const thead = el("thead");
  const headRow = el("tr");
  for (const heading of headers) headRow.appendChild(el("th", heading));
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = el("tbody");
  for (const cells of rows) {
    const row = el("tr");
    for (const cell of cells) {
      const td = el("td", cell && cell.text !== undefined ? cell.text : cell);
      if (cell && cell.className) td.className = cell.className;
      row.appendChild(td);
    }
    tbody.appendChild(row);
  }
  table.appendChild(tbody);
  return table;
}

function panelShell(panel, title, subtitle) {
  panel.replaceChildren();
  panel.hidden = false;
  panel.appendChild(el("h3", title, "resilience-card-title"));
  if (subtitle) panel.appendChild(el("p", subtitle, "resilience-card-note"));
  return panel;
}

function pct(value, digits = 2) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";
}

function renderRobustnessPanel(multiSeed) {
  const panel = byId("robustness-panel");
  if (!multiSeed || !multiSeed.aggregate) {
    panel.hidden = true;
    return;
  }
  const seeds = multiSeed.seeds || [];
  panelShell(
    panel,
    "Multi-seed robustness",
    `Full pipeline rerun on ${seeds.length} seeds (${seeds[0]}–${seeds.at(-1)}); ` +
      "intervals are two-sided 95% confidence.",
  );
  const agg = multiSeed.aggregate;
  const ciRow = (name, key) => [
    name,
    `${pct(agg[key].mean)} ± ${pct(agg[key].ci95_half_width)}`,
    pct(agg[key].min),
    pct(agg[key].max),
  ];
  panel.appendChild(
    buildTable(
      ["Metric", "Mean ± 95% CI", "Min", "Max"],
      [
        ciRow("Accuracy", "accuracy"),
        ciRow("Precision", "precision"),
        ciRow("Recall", "recall"),
        ciRow("F1 score", "f1_score"),
      ],
    ),
  );
  if (Array.isArray(multiSeed.per_seed) && multiSeed.per_seed.length) {
    panel.appendChild(el("h4", "Per-seed results", "resilience-subhead"));
    panel.appendChild(
      buildTable(
        ["Seed", "Accuracy", "Recall", "F1", "FP", "FN"],
        multiSeed.per_seed.map((run) => [
          run.seed,
          pct(run.accuracy),
          pct(run.recall),
          pct(run.f1_score),
          run.false_positives,
          run.false_negatives,
        ]),
      ),
    );
  }
}

function renderThresholdChart(sweep, panel) {
  const canvas = document.createElement("canvas");
  canvas.className = "threshold-canvas";
  canvas.width = 520;
  canvas.height = 260;
  canvas.setAttribute("role", "img");
  const description = sweep.points
    .map((point) => `sigma ${point.sigma}: ${pct(point.accuracy)}`)
    .join("; ");
  canvas.setAttribute("aria-label", `Accuracy by threshold sigma. ${description}.`);
  panel.appendChild(canvas);

  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const margin = { left: 52, right: 18, top: 18, bottom: 42 };
  const plotW = canvas.width - margin.left - margin.right;
  const plotH = canvas.height - margin.top - margin.bottom;
  const sigmas = sweep.points.map((point) => point.sigma);
  const minSigma = Math.min(...sigmas);
  const maxSigma = Math.max(...sigmas);
  const xFor = (sigma) => margin.left +
    ((sigma - minSigma) / Math.max(maxSigma - minSigma, 1)) * plotW;
  const yFor = (accuracy) => margin.top + (1 - accuracy) * plotH;

  ctx.fillStyle = "#081014";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (sweep.zero_error_band) {
    ctx.fillStyle = "rgba(54, 214, 199, 0.12)";
    const low = xFor(sweep.zero_error_band.low);
    const high = xFor(sweep.zero_error_band.high);
    ctx.fillRect(low, margin.top, Math.max(2, high - low), plotH);
  }
  ctx.strokeStyle = "#23363a";
  ctx.fillStyle = "#b8cbc7";
  ctx.font = "12px ui-monospace, monospace";
  for (let i = 0; i <= 4; i += 1) {
    const accuracy = i / 4;
    const y = yFor(accuracy);
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
    ctx.fillText(accuracy.toFixed(2), 12, y + 4);
  }
  ctx.strokeStyle = "#36d6c7";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  sweep.points.forEach((point, index) => {
    const x = xFor(point.sigma);
    const y = yFor(point.accuracy);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#b8cbc7";
  ctx.fillText(String(minSigma), margin.left - 6, canvas.height - 16);
  ctx.fillText(String(maxSigma), margin.left + plotW - 12, canvas.height - 16);
  ctx.fillText("Threshold sigma", margin.left + plotW / 2 - 48, canvas.height - 8);
}

function renderThresholdPanel(sweep) {
  const panel = byId("threshold-panel");
  if (!sweep || !Array.isArray(sweep.points)) {
    panel.hidden = true;
    return;
  }
  const band = sweep.zero_error_band;
  panelShell(
    panel,
    "Threshold sensitivity",
    band
      ? `Zero-error band spans sigma ${band.low}–${band.high}; the mandated ` +
          `${sweep.configured_sigma} sits inside a stable plateau.`
      : "No sigma achieved zero errors on this run.",
  );
  renderThresholdChart(sweep, panel);
  const rows = sweep.points.map((point) => {
    const clean = point.false_positives === 0 && point.false_negatives === 0;
    return [
      {
        text: point.sigma === sweep.configured_sigma
          ? `${point.sigma} ◂ configured`
          : String(point.sigma),
        className: point.sigma === sweep.configured_sigma ? "is-configured" : "",
      },
      pct(point.accuracy),
      { text: point.false_positives, className: point.false_positives ? "is-warn" : "" },
      { text: point.false_negatives, className: point.false_negatives ? "is-warn" : "" },
      { text: clean ? "clean" : "errors", className: clean ? "is-ok" : "is-warn" },
    ];
  });
  panel.appendChild(buildTable(["Sigma", "Accuracy", "FP", "FN", "State"], rows));
}

function renderRocPanel(roc) {
  const panel = byId("roc-panel");
  const canvas = byId("roc-canvas");
  if (!roc || !roc.reconstruction || !canvas) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  setText(
    "roc-note",
    `Threshold-independent ranking quality. Reconstruction AUC ` +
      `${roc.reconstruction.auc.toFixed(3)}; fused (recon + THD) AUC ` +
      `${roc.combined.auc.toFixed(3)}. Diagonal is chance.`,
  );
  setText(
    "roc-a11y",
    `ROC curves. Reconstruction-error area under curve ` +
      `${roc.reconstruction.auc.toFixed(3)}, fused score area under curve ` +
      `${roc.combined.auc.toFixed(3)}. A perfect detector reaches the ` +
      `top-left corner.`,
  );

  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const margin = { left: 54, right: 16, top: 16, bottom: 44 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#081014";
  ctx.fillRect(0, 0, width, height);

  const xFor = (fpr) => margin.left + fpr * plotW;
  const yFor = (tpr) => margin.top + (1 - tpr) * plotH;

  ctx.strokeStyle = "#23363a";
  ctx.fillStyle = "#9fb6b0";
  ctx.font = "12px ui-monospace, monospace";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const gx = margin.left + (plotW * i) / 5;
    const gy = margin.top + (plotH * i) / 5;
    ctx.beginPath();
    ctx.moveTo(gx, margin.top);
    ctx.lineTo(gx, margin.top + plotH);
    ctx.moveTo(margin.left, gy);
    ctx.lineTo(margin.left + plotW, gy);
    ctx.stroke();
    ctx.fillText((i / 5).toFixed(1), gx - 8, height - 26);
    ctx.fillText((1 - i / 5).toFixed(1), 20, gy + 4);
  }
  ctx.fillText("False-positive rate", margin.left + plotW / 2 - 60, height - 8);

  ctx.strokeStyle = "#4a5f66";
  ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(xFor(0), yFor(0));
  ctx.lineTo(xFor(1), yFor(1));
  ctx.stroke();
  ctx.setLineDash([]);

  const drawCurve = (curve, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    curve.fpr.forEach((fpr, i) => {
      const x = xFor(fpr);
      const y = yFor(curve.tpr[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  drawCurve(roc.reconstruction, "#36d6c7");
  drawCurve(roc.combined, "#ffb84d");

  if (roc.operating_point) {
    ctx.fillStyle = "#eef7f3";
    ctx.beginPath();
    ctx.arc(xFor(roc.operating_point.fpr), yFor(roc.operating_point.tpr), 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function renderResiliencePanel(data) {
  const panel = byId("resilience-panel");
  const probes = data.boundary_probes;
  const edge = data.edge_benchmark;
  const drift = data.drift_monitor;
  const fdi = data.fdi_resilience;
  if (!probes && !edge && !drift && !fdi) {
    panel.hidden = true;
    return;
  }
  panelShell(
    panel,
    "Adversarial & operational resilience",
    "Boundary probes, edge quantization, drift, and false-data-injection.",
  );

  if (probes && Array.isArray(probes.probes)) {
    panel.appendChild(el("h4", "Decision-boundary probes", "resilience-subhead"));
    panel.appendChild(
      buildTable(
        ["Probe", "Prediction", "Result"],
        probes.probes.map((probe) => [
          probe.name.replace(/_/g, " "),
          probe.observed.prediction,
          {
            text: probe.pass ? "pass" : "FAIL",
            className: probe.pass ? "is-ok" : "is-alarm",
          },
        ]),
      ),
    );
  }

  if (fdi && Array.isArray(fdi.attacks)) {
    panel.appendChild(el("h4", "False-data-injection resilience", "resilience-subhead"));
    panel.appendChild(
      el(
        "p",
        `Baseline false positives on ${fdi.baseline_records} genuine records: ` +
          `${fdi.baseline_false_positives}.`,
        "resilience-card-note",
      ),
    );
    panel.appendChild(
      buildTable(
        ["Attack", "Evaded learned", "Caught after evasion", "Residual", "Conditional"],
        fdi.attacks.map((attack) => [
          attack.name.replace(/_/g, " "),
          attack.evaded_learned_detector,
          attack.caught_after_learned_evasion,
          attack.residual_evasions,
          {
            text: pct(attack.detection_rate_among_evasions, 1),
            className: attack.detection_rate_among_evasions >= 0.95
              ? "is-ok"
              : "is-warn",
          },
        ]),
      ),
    );
    if (Number.isFinite(fdi.coordinated_evades_learned_detector)) {
      const coordinated = fdi.attacks.find((attack) => attack.name === "coordinated");
      const passed = Boolean(fdi.coordinated_layered_defense_proven);
      panel.appendChild(
        el(
          "p",
          passed && coordinated
            ? `Layered-defense gate passed: the coordinated attack evaded the ` +
              `learned detector on ${pct(fdi.coordinated_evades_learned_detector, 0)} ` +
              `of records; physics caught ${pct(
                coordinated.detection_rate_among_evasions, 1,
              )} of those evasions, leaving ${coordinated.residual_evasions}.`
            : "The coordinated-attack layered-defense gate did not pass.",
          `resilience-card-note ${passed ? "" : "is-warn"}`.trim(),
        ),
      );
    }
  }

  if (edge && edge.fp32) {
    panel.appendChild(el("h4", "Autoencoder edge benchmark", "resilience-subhead"));
    if (edge.int8) {
      panel.appendChild(
        buildTable(
          ["Variant", "State dict", "AE forward latency", "Accuracy"],
          [
            [
              "FP32",
              `${edge.fp32.state_dict_bytes.toLocaleString()} B`,
              `${edge.fp32.single_record_latency_ms_median.toFixed(2)} ms`,
              pct(edge.fp32.accuracy),
            ],
            [
              "int8",
              `${edge.int8.state_dict_bytes.toLocaleString()} B`,
              `${edge.int8.single_record_latency_ms_median.toFixed(2)} ms`,
              pct(edge.int8.accuracy),
            ],
          ],
        ),
      );
      panel.appendChild(
        el(
          "p",
          `Quantization shrinks the state dict ${edge.size_reduction_ratio.toFixed(2)}×.`,
          "resilience-card-note",
        ),
      );
    } else {
      panel.appendChild(
        el("p", `int8 unavailable: ${edge.int8_unavailable_reason}`, "resilience-card-note"),
      );
    }
  }

  if (drift && Array.isArray(drift.gain_scenarios)) {
    panel.appendChild(el("h4", "Drift monitoring", "resilience-subhead"));
    panel.appendChild(
      buildTable(
        ["Gain", "Individual alerts", "Max |z|", "Drift alarm"],
        drift.gain_scenarios.map((scenario) => [
          scenario.gain,
          scenario.individual_alerts,
          scenario.max_abs_z.toFixed(2),
          {
            text: scenario.silent_drift_detected
              ? "silent drift caught"
              : scenario.first_drift_alarm_at ?? "never",
            className: scenario.silent_drift_detected ? "is-ok" : "",
          },
        ]),
      ),
    );
  }

  const latency = data.latency_profile;
  if (latency && Array.isArray(latency.points) && latency.points.length) {
    panel.appendChild(el("h4", "Latency & throughput profile", "resilience-subhead"));
    panel.appendChild(
      buildTable(
        ["Batch", "Latency (ms)", "Per-record (ms)", "Throughput /s"],
        latency.points.map((point) => [
          {
            text: point.batch_size === latency.peak_throughput_batch_size
              ? `${point.batch_size} ◂ peak`
              : String(point.batch_size),
            className: point.batch_size === latency.peak_throughput_batch_size
              ? "is-ok"
              : "",
          },
          point.latency_ms_median.toFixed(2),
          point.per_record_ms.toFixed(3),
          Math.round(point.throughput_per_s).toLocaleString(),
        ]),
      ),
    );
    panel.appendChild(
      el(
        "p",
        `Measured on ${latency.device}. Small batches minimize per-alert ` +
          `latency; large batches maximize throughput.`,
        "resilience-card-note",
      ),
    );
  }
}

function renderDispatchPanel() {
  const panel = byId("dispatch-panel");
  const artifact = state.dispatches;
  if (!artifact || !Array.isArray(artifact.operational_dispatches)) {
    if (state.dispatchError) {
      panelShell(
        panel,
        "Self-healing dispatch unavailable",
        `The dispatch artifact could not be verified: ${state.dispatchError}`,
      );
      return;
    }
    panel.hidden = true;
    return;
  }
  const dispatches = artifact.operational_dispatches;
  panelShell(
    panel,
    "Self-healing dispatch",
    `${dispatches.length} prioritized work orders across a synthetic ` +
      `${(artifact.geospatial_reference && "8") || "?"}-section feeder. ` +
      "Simulated response, not a live control system.",
  );
  const priorityClass = (priority) =>
    priority === "CRITICAL" ? "is-alarm" : priority === "HIGH" ? "is-warn" : "";
  if (artifact.detected_assets.length) {
    panel.appendChild(el("h4", "Detected assets", "resilience-subhead"));
    panel.appendChild(
      buildTable(
        ["Asset", "Section", "Confidence", "Anomaly"],
        artifact.detected_assets.slice(0, 24).map((asset) => [
          asset.asset_id,
          asset.section,
          pct(asset.confidence_score, 1),
          asset.detected_anomalies.join(", ").replace(/_/g, " "),
        ]),
      ),
    );
  }
  panel.appendChild(el("h4", "Operational work orders", "resilience-subhead"));
  panel.appendChild(
    buildTable(
      ["Dispatch", "Node", "Priority", "Action"],
      dispatches.slice(0, 24).map((dispatch) => [
        dispatch.dispatch_id,
        dispatch.target_node,
        { text: dispatch.priority, className: priorityClass(dispatch.priority) },
        dispatch.action_required,
      ]),
    ),
  );
  if (artifact.simulation_disclaimer) {
    panel.appendChild(el("p", artifact.simulation_disclaimer, "resilience-card-note"));
  }
}

async function loadDispatches(runId, resultsSha256) {
  if (
    state.dispatches && dispatchCacheMatches(
      state.dispatchRunId, state.dispatchResultsSha256, runId, resultsSha256,
    )
  ) return;
  const requestId = ++state.dispatchRequestId;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const headers = {};
    if (state.dispatchValidators.etag) {
      headers["If-None-Match"] = state.dispatchValidators.etag;
    }
    if (state.dispatchValidators.lastModified) {
      headers["If-Modified-Since"] = state.dispatchValidators.lastModified;
    }
    const response = await fetch(DISPATCHES_URL, {
      cache: "no-store",
      credentials: "same-origin",
      headers,
      signal: controller.signal,
    });
    if (response.status === 304 && state.dispatches) return;
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await readBodyWithCap(response, MAX_RESULTS_BYTES);
    const artifact = assertDispatchShape(
      JSON.parse(text), runId, resultsSha256,
    );
    if (
      requestId !== state.dispatchRequestId ||
      state.data?.run?.id !== runId || state.resultsSha256 !== resultsSha256
    ) return;
    state.dispatches = artifact;
    state.dispatchError = null;
    state.dispatchRunId = runId;
    state.dispatchResultsSha256 = resultsSha256;
    state.dispatchValidators.etag = response.headers.get("etag");
    state.dispatchValidators.lastModified = response.headers.get("last-modified");
  } catch (error) {
    if (requestId === state.dispatchRequestId) {
      state.dispatches = null; // Panel simply stays hidden if unavailable.
      state.dispatchError = error instanceof Error ? error.message : "unknown error";
      state.dispatchRunId = null;
      state.dispatchResultsSha256 = null;
    }
  } finally {
    clearTimeout(timer);
    if (requestId !== state.dispatchRequestId) return;
    renderDispatchPanel();
    byId("resilience-section").hidden = ![
      "robustness-panel",
      "threshold-panel",
      "roc-panel",
      "resilience-panel",
      "dispatch-panel",
      "grid-response-panel",
    ].some((id) => !byId(id).hidden);
  }
}

function renderGridResponsePanel(gridResponse) {
  const panel = byId("grid-response-panel");
  if (!gridResponse) {
    panel.hidden = true;
    return;
  }
  panelShell(
    panel,
    "Grid response summary",
    `Simulated self-healing across ${gridResponse.sections} feeder sections. ` +
      "Not a live control system.",
  );
  const stats = [
    { label: "Alerts dispatched", value: gridResponse.alerts_dispatched, cls: "is-cyan" },
    { label: "Sections isolated", value: gridResponse.sections_isolated.length, cls: "is-amber" },
    { label: "Customers affected", value: gridResponse.customers_in_isolated_sections, cls: "is-warn" },
    { label: "Restored via reroute", value: gridResponse.customers_restored_via_reroute, cls: "is-green" },
    { label: "Isolation time", value: `${gridResponse.simulated_isolation_seconds}s`, cls: "" },
    { label: "Reroute time", value: `${gridResponse.simulated_reroute_seconds}s`, cls: "" },
  ];
  const row = el("div", null, "resilience-stat-row");
  for (const stat of stats) {
    const card = el("div", null, "resilience-stat");
    card.appendChild(el("span", stat.label));
    card.appendChild(el("strong", String(stat.value), stat.cls));
    row.appendChild(card);
  }
  panel.appendChild(row);
  const priorities = gridResponse.priorities || {};
  const prioEntries = Object.entries(priorities);
  if (prioEntries.length) {
    panel.appendChild(el("h4", "Priority breakdown", "resilience-subhead"));
    const priorityClass = (p) =>
      p === "CRITICAL" ? "is-alarm" : p === "HIGH" ? "is-warn" : "";
    panel.appendChild(
      buildTable(
        ["Priority", "Count"],
        prioEntries.map(([k, v]) => [
          { text: k, className: priorityClass(k) },
          String(v),
        ]),
      ),
    );
  }
  if (gridResponse.sections_isolated && gridResponse.sections_isolated.length) {
    panel.appendChild(el("h4", "Isolated sections", "resilience-subhead"));
    panel.appendChild(
      el("p", gridResponse.sections_isolated.join(", "), "resilience-card-note"),
    );
  }
}

function renderResilience(data) {
  const section = byId("resilience-section");
  if (!dispatchCacheMatches(
    state.dispatchRunId,
    state.dispatchResultsSha256,
    data.run.id,
    state.resultsSha256,
  )) {
    state.dispatches = null;
    state.dispatchError = null;
    state.dispatchRunId = null;
    state.dispatchResultsSha256 = null;
  }
  renderRobustnessPanel(data.multi_seed);
  renderThresholdPanel(data.threshold_sweep);
  renderRocPanel(data.roc_analysis);
  renderResiliencePanel(data);
  renderDispatchPanel();
  renderGridResponsePanel(data.grid_response);
  const anyVisible = [
    "robustness-panel",
    "threshold-panel",
    "roc-panel",
    "resilience-panel",
    "dispatch-panel",
    "grid-response-panel",
  ].some((id) => !byId(id).hidden);
  section.hidden = !anyVisible;
  loadDispatches(data.run.id, state.resultsSha256);
  loadAdversarialResilience();
}

// --- Adversarial resilience panel ---

async function loadAdversarialResilience() {
  const intro = byId("adversarial-intro");
  const summary = byId("adversarial-summary");
  const grid = byId("adversarial-grid");
  try {
    const response = await fetch("adversarial_resilience.json", {
      cache: "no-store",
      credentials: "same-origin",
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderAdversarialResilience(data);
  } catch (error) {
    intro.textContent = "Adversarial resilience data unavailable.";
    console.error("Adversarial resilience fetch failed:", error);
  }
}

function renderAdversarialResilience(data) {
  const intro = byId("adversarial-intro");
  const summary = byId("adversarial-summary");
  const grid = byId("adversarial-grid");

  intro.textContent =
    `${data.total_scenarios} attack and sensor-failure scenarios tested across ` +
    `${data.test_suite_size} unit tests. Each card shows the detection mechanism ` +
    `that catches (or allows) the scenario.`;

  // Summary stats
  summary.hidden = false;
  summary.replaceChildren();
  const stats = [
    { label: "Scenarios tested", value: data.total_scenarios, cls: "is-cyan" },
    { label: "Correctly handled", value: data.flagged_count + data.passed_count, cls: "is-green" },
    { label: "Flagged (attacked)", value: data.flagged_count, cls: "is-amber" },
    { label: "All correct", value: data.all_correct ? "Yes" : "No", cls: "is-green" },
  ];
  for (const stat of stats) {
    const card = el("div", null, "adversarial-stat");
    card.appendChild(el("span", stat.label));
    const strong = el("strong", String(stat.value), stat.cls);
    card.appendChild(strong);
    summary.appendChild(card);
  }

  // Scenario cards
  grid.replaceChildren();
  for (const scenario of data.scenarios) {
    const card = el("div", null, "adversarial-card");

    // Header: title + badge
    const header = el("div", null, "adversarial-card-header");
    header.appendChild(el("h3", scenario.name));
    const isFlagged = scenario.observed_detection === "flagged";
    const badgeClass = isFlagged ? "adversarial-badge is-flagged" : "adversarial-badge is-passed";
    header.appendChild(el("span", isFlagged ? "Detected" : "Passed", badgeClass));
    card.appendChild(header);

    // Category tag
    card.appendChild(el("span", scenario.category, "adversarial-card-category"));

    // Description
    card.appendChild(el("p", scenario.description));

    // Detection mechanism
    const mech = el("div", null, "mechanism");
    mech.textContent = scenario.detection_mechanism;
    card.appendChild(mech);

    grid.appendChild(card);
  }
}

function renderCurrentRecord() {
  const record = state.records[state.current];
  if (!record || !state.data) return;
  const calibration = state.data.training.calibration;
  const thdLimit = state.data.config.thd_limit;
  const isAnomaly = record.prediction === "anomaly";

  setText("sample-index", `Test ${record.sample_index}`);
  setText("sample-type", record.anomaly_type.replaceAll("_", " "));
  setText("record-position", `${state.current + 1} / ${state.records.length}`);
  byId("record-slider").value = String(state.current);
  setText("prediction", record.prediction);
  setText("ground-truth", record.ground_truth);
  setText("source-index", record.source_index);
  setText("reconstruction-value", fixed(record.reconstruction_error, 8));
  setText("thd-value", evidencePercent(record.thd_ratio));

  const decision = byId("decision-status");
  decision.dataset.state = isAnomaly ? "anomaly" : "normal";
  const reconstructionRatio = evidenceRatio(record.reconstruction_error, calibration.threshold);
  const thdRatio = evidenceRatio(record.thd_ratio, thdLimit);
  const reconstructionMeter = byId("reconstruction-meter");
  const thdMeter = byId("thd-meter");
  reconstructionMeter.max = 2;
  thdMeter.max = 2;
  reconstructionMeter.value = Math.min(reconstructionRatio, 2);
  thdMeter.value = Math.min(thdRatio, 2);

  const triggerList = byId("trigger-list");
  triggerList.replaceChildren();
  const labels = {
    invalid_input: "Invalid or non-finite telemetry",
    reconstruction_error: "Reconstruction error crossed its calibrated limit",
    thd: "THD crossed the 5% power-quality limit",
  };
  const triggers = record.triggers.length ? record.triggers : ["none"];
  for (const trigger of triggers) {
    const item = document.createElement("li");
    item.textContent = labels[trigger] ?? "No threshold crossed";
    triggerList.appendChild(item);
  }
  setText(
    "waveform-summary",
    `Test record ${record.sample_index} is ${record.anomaly_type}. Ground truth is ${record.ground_truth}; prediction is ${record.prediction}. Reconstruction error is ${fixed(record.reconstruction_error, 8)} and THD is ${evidencePercent(record.thd_ratio)}.`,
  );
  if (liveState.enabled) {
    drawWaveformWithFault();
    applyLiveSimulation();
  } else {
    drawWaveform(record);
  }
}

function drawWaveform(record) {
  const canvas = byId("waveform-canvas");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const margin = { left: 64, right: 24, top: 24, bottom: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const colors = ["#36d6c7", "#7aa8ff", "#ffb84d"];
  const pointCount = state.data.replay.time_ms.length;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#081014";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "#23363a";
  context.fillStyle = "#9fb6b0";
  context.font = "14px ui-monospace, monospace";
  context.lineWidth = 1;
  for (let index = 0; index <= 6; index += 1) {
    const x = margin.left + (plotWidth * index) / 6;
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, margin.top + plotHeight);
    context.stroke();
    const time = state.data.replay.time_ms[Math.round(((pointCount - 1) * index) / 6)];
    context.fillText(`${fixed(time, 1)}`, x - 14, height - 18);
  }
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + (plotHeight * index) / 4;
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();
    context.fillText((1.5 - index * 0.75).toFixed(2), 10, y + 5);
  }

  const yFor = (value) => margin.top + plotHeight / 2 - (finiteNumber(value) / 1.5) * (plotHeight / 2);
  for (let phase = 0; phase < 3; phase += 1) {
    const actual = record.actual[phase];
    const reconstruction = record.reconstruction[phase];
    context.strokeStyle = colors[phase];
    context.lineWidth = 2.5;
    context.setLineDash([]);
    context.beginPath();
    actual.forEach((value, index) => {
      const x = margin.left + (plotWidth * index) / Math.max(actual.length - 1, 1);
      const y = yFor(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();

    context.strokeStyle = "#9fb6b0";
    context.lineWidth = 1.2;
    context.setLineDash([7, 7]);
    context.beginPath();
    reconstruction.forEach((value, index) => {
      const x = margin.left + (plotWidth * index) / Math.max(reconstruction.length - 1, 1);
      const y = yFor(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
  context.setLineDash([]);
  const cursorX = margin.left + plotWidth * state.scan;
  context.strokeStyle = "#eef7f3";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(cursorX, margin.top);
  context.lineTo(cursorX, margin.top + plotHeight);
  context.stroke();
}

function renderTrainingCurve() {
  const canvas = byId("training-canvas");
  const context = canvas.getContext("2d");
  const history = state.data.training.epochs;
  const width = canvas.width;
  const height = canvas.height;
  const margin = { left: 54, right: 20, top: 20, bottom: 42 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#081014";
  context.fillRect(0, 0, width, height);
  if (!history.length) return;
  const losses = history.map((item) => Math.log10(Math.max(item.mse, 1e-12)));
  const low = Math.min(...losses);
  const high = Math.max(...losses);
  context.strokeStyle = "#23363a";
  context.fillStyle = "#9fb6b0";
  context.font = "13px ui-monospace, monospace";
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + (plotHeight * index) / 4;
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
  }
  context.strokeStyle = "#36d6c7";
  context.lineWidth = 3;
  context.beginPath();
  losses.forEach((loss, index) => {
    const x = margin.left + (plotWidth * index) / Math.max(losses.length - 1, 1);
    const y = margin.top + plotHeight - ((loss - low) / Math.max(high - low, 1e-9)) * plotHeight;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
  history.forEach((item, index) => {
    const x = margin.left + (plotWidth * index) / Math.max(history.length - 1, 1);
    const y = margin.top + plotHeight - ((losses[index] - low) / Math.max(high - low, 1e-9)) * plotHeight;
    context.fillStyle = "#eef7f3";
    context.beginPath();
    context.arc(x, y, 3.5, 0, Math.PI * 2);
    context.fill();
    if (index === 0 || index === history.length - 1) {
      context.fillStyle = "#9fb6b0";
      context.fillText(`E${item.epoch}`, x - 9, height - 16);
    }
  });
}

function renderAlerts() {
  const body = byId("alerts-body");
  body.replaceChildren();
  const alerts = state.data.evaluation.observations.filter(
    (item) => item.prediction === "anomaly",
  );
  setText("alert-count", `${alerts.length} / ${state.data.dataset.test_total} flagged`);
  for (const alert of alerts) {
    const row = document.createElement("tr");
    const indexCell = document.createElement("td");
    const button = document.createElement("button");
    button.className = "sample-button";
    button.type = "button";
    button.textContent = String(alert.index);
    button.setAttribute("aria-label", `Replay test record ${alert.index}`);
    button.addEventListener("click", () => {
      const replayIndex = state.records.findIndex((item) => item.sample_index === alert.index);
      if (replayIndex >= 0) selectRecord(replayIndex);
      else setText("refresh-status", `Record ${alert.index} has scalar evidence but no replay trace.`);
    });
    indexCell.appendChild(button);
    row.appendChild(indexCell);
    for (const value of [
      alert.anomaly_type,
      alert.ground_truth,
      fixed(alert.reconstruction_error, 8),
      evidencePercent(alert.thd_ratio),
      alert.triggers.join(" + ") || "none",
      explainAlert(alert.explanation),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = String(value);
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
}

function explainAlert(explanation) {
  if (!explanation || explanation.dominant_harmonic === null) return "—";
  const harmonicShare =
    explanation.low_harmonic_fraction + explanation.high_harmonic_fraction;
  if (harmonicShare < 0.02) {
    return `non-harmonic · phase ${explanation.worst_phase}`;
  }
  return `H${explanation.dominant_harmonic} · phase ${explanation.worst_phase} · ` +
    `${(harmonicShare * 100).toFixed(0)}% harmonic`;
}

function selectRecord(index) {
  if (!state.records.length) return;
  state.current = (index + state.records.length) % state.records.length;
  state.scan = 0;
  renderCurrentRecord();
  renderSimClock();
}

function updatePlayButton() {
  const button = byId("play-button");
  button.setAttribute("aria-pressed", String(state.playing));
  button.textContent = state.playing ? "Pause replay" : "Play replay";
}

function renderSimClock() {
  const timeAxis = state.data?.replay?.time_ms;
  const windowMs = timeAxis && timeAxis.length ? timeAxis[timeAxis.length - 1] : 0;
  const signalMs = state.scan * windowMs;
  setText("sim-time", `${signalMs.toFixed(2)} ms`);
  setText("sim-cycles", (signalMs / (1000 / 60)).toFixed(2));
  setText("sim-tick", simClock.tick.toLocaleString());
  setText(
    "sim-mode",
    simClock.everyFrame ? "Every frame (decoupled)" : "Real-time (fixed step)",
  );
  setText(
    "sim-dropped",
    simClock.droppedTicks ? simClock.droppedTicks.toLocaleString() : "0",
  );
}

function stepTicks(count) {
  if (!state.records.length) return;
  state.scan += count / simClock.ticksPerSweep;
  if (state.scan >= 1) selectRecord(state.current + 1);
  else drawWaveform(state.records[state.current]);
  renderSimClock();
}

let animationFrameId = null;

function requestAnimationLoop() {
  if (animationFrameId === null && !document.hidden) {
    animationFrameId = requestAnimationFrame(animate);
  }
}

function animate(timestamp) {
  animationFrameId = null;
  const wallDeltaSeconds = Math.min((timestamp - state.lastFrame) / 1000, 0.1);
  state.lastFrame = timestamp;
  if (state.playing && state.records.length) {
    const ticks = simClock.advance(wallDeltaSeconds, state.speed);
    if (ticks > 0) stepTicks(ticks);
  }
  if (window.GridTopology) window.GridTopology.frame();
  // Update DSP engine at 60fps (always, even when paused)
  updateDspEngine();
  // Keep animation running for DSP updates
  requestAnimationLoop();
}

byId("play-button").addEventListener("click", () => {
  state.playing = !state.playing;
  updatePlayButton();
  if (state.playing) {
    state.lastFrame = performance.now();
    requestAnimationLoop();
  }
});
byId("step-button").addEventListener("click", () => {
  if (state.playing) {
    state.playing = false;
    updatePlayButton();
  }
  simClock.tick += 1;
  stepTicks(1);
});
byId("everyframe-toggle").addEventListener("change", (event) => {
  simClock.everyFrame = event.target.checked;
  simClock.accumulator = 0;
  renderSimClock();
});
byId("speed-select").addEventListener("change", (event) => {
  state.speed = finiteNumber(Number(event.target.value), 1);
});
byId("refresh-button").addEventListener("click", () => {
  state.pollDelay = POLL_BASE_MS;
  loadResults();
});
byId("previous-button").addEventListener("click", () => selectRecord(state.current - 1));
byId("next-button").addEventListener("click", () => selectRecord(state.current + 1));
byId("record-slider").addEventListener("input", (event) => {
  selectRecord(Number(event.target.value));
});

// Keyboard navigation for records: left/right when waveform area is focused
document.addEventListener("keydown", (event) => {
  const target = event.target;
  if (target.tagName === "SELECT" || target.tagName === "INPUT" || target.tagName === "BUTTON") return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    selectRecord(state.current - 1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    selectRecord(state.current + 1);
  } else if (event.key === " ") {
    event.preventDefault();
    state.playing = !state.playing;
    updatePlayButton();
    if (state.playing) {
      state.lastFrame = performance.now();
      requestAnimationLoop();
    }
  }
});

reducedMotion.addEventListener("change", (event) => {
  if (event.matches) {
    state.playing = false;
    updatePlayButton();
  }
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    state.pollDelay = POLL_BASE_MS;
    loadResults({ announce: false });
    if (state.playing) {
      state.lastFrame = performance.now();
      requestAnimationLoop();
    }
  }
});

// --- Self-contained HTML report generation ---

async function generateReport() {
  const data = state.data;
  if (!data) return;
  const metrics = data.evaluation.metrics;
  const calibration = data.training.calibration;
  const training = data.training.epochs;
  const completed = new Date(data.run.completed_at_utc);
  const completedStr = Number.isNaN(completed.getTime()) ? "n/a" : completed.toLocaleString();
  const finalLoss = training.length ? training.at(-1).mse.toFixed(8) : "n/a";
  const alerts = data.evaluation.observations.filter((item) => item.prediction === "anomaly");
  const alertRows = alerts.map((a) =>
    `<tr><td>${a.index}</td><td>${a.anomaly_type}</td><td>${a.ground_truth}</td>` +
    `<td>${a.reconstruction_error.toFixed(8)}</td>` +
    `<td>${(a.thd_ratio * 100).toFixed(2)}%</td>` +
    `<td>${a.triggers.join(", ") || "none"}</td></tr>`
  ).join("\n");

  let advHtml = "<p>Adversarial resilience data unavailable.</p>";
  try {
    const advResp = await fetch("adversarial_resilience.json", { cache: "no-store" });
    if (advResp.ok) {
      const adv = await advResp.json();
      const advRows = adv.scenarios.map((s) =>
        `<tr><td>${s.name}</td><td>${s.category}</td>` +
        `<td style="color:${s.observed_detection === "flagged" ? "#f85149" : "#3fb950"}">${s.observed_detection}</td>` +
        `<td>${s.detection_mechanism}</td></tr>`
      ).join("\n");
      advHtml = `<p>${adv.total_scenarios} scenarios tested. All correct: ${adv.all_correct ? "Yes" : "No"}.</p>` +
        `<table><thead><tr><th>Scenario</th><th>Category</th><th>Detection</th><th>Mechanism</th></tr></thead><tbody>${advRows}</tbody></table>`;
    }
  } catch (_) { /* keep fallback */ }
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Grid Sentinel Report — ${data.run.id}</title>
<style>
  :root { color-scheme: dark; --bg: #0e1117; --surface: #161b22; --border: #30363d; --ink: #e6edf3; --muted: #8b949e; --cyan: #58a6ff; --green: #3fb950; --red: #f85149; --amber: #d29922; }
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: var(--bg); color: var(--ink); padding: 2rem; line-height: 1.6; }
  h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.15rem; margin: 1.8rem 0 0.6rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
  .subtitle { color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; }
  .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.8rem; text-align: center; }
  .stat span { display: block; color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; }
  .stat strong { display: block; margin-top: 0.2rem; font-size: 1.3rem; color: var(--cyan); }
  table { width: 100%; border-collapse: collapse; margin-top: 0.5rem; font-size: 0.82rem; }
  th, td { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.03em; }
  dl { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.6rem; }
  div { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 0.6rem 0.8rem; }
  dt { color: var(--muted); font-size: 0.72rem; }
  dd { margin: 0.15rem 0 0; font-family: monospace; font-size: 0.82rem; }
  .limitation { margin-top: 1.2rem; padding: 0.7rem 0.9rem; border: 1px solid rgba(210,153,34,0.3); border-radius: 6px; background: rgba(210,153,34,0.08); color: var(--amber); font-size: 0.82rem; }
  .footer { margin-top: 2rem; padding-top: 0.8rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.72rem; }
</style>
</head>
<body>
<h1>Grid Sentinel — Simulation Report</h1>
<p class="subtitle">Self-contained export from recorded run ${data.run.id}</p>

<h2>Run Metadata</h2>
<dl>
  <div><dt>Completed</dt><dd>${completedStr}</dd></div>
  <div><dt>Seed</dt><dd>${data.config.seed}</dd></div>
  <div><dt>Device</dt><dd>${data.model.device}</dd></div>
  <div><dt>Model</dt><dd>${data.model.name} (${data.model.parameters.toLocaleString()} params)</dd></div>
  <div><dt>Source hash</dt><dd>${data.provenance.source_sha256}</dd></div>
  <div><dt>Runtime</dt><dd>Python ${data.provenance.python_version} · torch ${data.provenance.torch_version}</dd></div>
</dl>

<h2>Validation Metrics</h2>
<div class="grid">
  <div class="stat"><span>Accuracy</span><strong>${(metrics.accuracy * 100).toFixed(2)}%</strong></div>
  <div class="stat"><span>Precision</span><strong>${(metrics.precision * 100).toFixed(2)}%</strong></div>
  <div class="stat"><span>Recall</span><strong>${(metrics.recall * 100).toFixed(2)}%</strong></div>
  <div class="stat"><span>F1 Score</span><strong>${(metrics.f1_score * 100).toFixed(2)}%</strong></div>
  <div class="stat"><span>Inference Latency</span><strong>${metrics.inference_latency_ms.toFixed(2)} ms</strong></div>
  <div class="stat"><span>Throughput</span><strong>${metrics.throughput_waveforms_per_second.toFixed(1)} /s</strong></div>
</div>

<h2>Confusion Matrix</h2>
<table>
  <thead><tr><th>TP</th><th>TN</th><th>FP</th><th>FN</th></tr></thead>
  <tbody><tr><td>${metrics.true_positives}</td><td>${metrics.true_negatives}</td><td>${metrics.false_positives}</td><td>${metrics.false_negatives}</td></tr></tbody>
</table>

<h2>Training</h2>
<dl>
  <div><dt>Final loss</dt><dd>${finalLoss}</dd></div>
  <div><dt>Calibration mean</dt><dd>${calibration.mean_error.toFixed(8)}</dd></div>
  <div><dt>Calibration σ</dt><dd>${calibration.std_error.toFixed(8)}</dd></div>
  <div><dt>Threshold</dt><dd>${calibration.threshold.toFixed(8)}</dd></div>
  <div><dt>THD limit</dt><dd>${(data.config.thd_limit * 100).toFixed(2)}%</dd></div>
  <div><dt>Epochs</dt><dd>${training.length}</dd></div>
</dl>

<h2>Alert Ledger (${alerts.length} flagged)</h2>
<table>
  <thead><tr><th>Index</th><th>Type</th><th>Truth</th><th>Recon. error</th><th>THD</th><th>Triggers</th></tr></thead>
  <tbody>${alertRows || "<tr><td colspan='6'>No alerts</td></tr>"}</tbody>
</table>

<h2>Adversarial Resilience</h2>
${advHtml}

<div class="limitation">
  Scope note: perfect performance on a seeded synthetic dataset is a pipeline
  verification result, not a claim of utility-field accuracy.
</div>
<p class="footer">Generated from Grid Sentinel recorded replay. All data is deterministic and reproducible.</p>
</body>
</html>`;

  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `grid-sentinel-report-${data.run.id}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Enable download button when data loads
const downloadButton = byId("download-report");
if (downloadButton) {
  downloadButton.addEventListener("click", generateReport);
}

// --- Multi-run comparison ---

async function loadComparison() {
  const altUrl = RUN_MANIFEST.find((r) => r.url !== RESULTS_URL)?.url;
  if (!altUrl) return;
  try {
    const response = await fetch(altUrl, { cache: "no-store", credentials: "same-origin" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await readBodyWithCap(response, MAX_RESULTS_BYTES);
    state.comparisonData = JSON.parse(text);
    renderComparison();
  } catch (error) {
    console.error("Comparison run fetch failed:", error);
    state.comparisonData = null;
  }
}

function renderComparison() {
  const section = byId("comparison-section");
  const alt = state.comparisonData;
  const primary = state.data;
  if (!alt || !primary) {
    section.hidden = true;
    return;
  }
  section.hidden = false;

  byId("comparison-heading").textContent =
    `Seed ${primary.config.seed} vs Seed ${alt.config.seed}`;

  const trainingEl = byId("compare-training");
  trainingEl.replaceChildren();
  const pEpochs = primary.training.epochs;
  const aEpochs = alt.training.epochs;
  const pFinalLoss = pEpochs.length ? pEpochs.at(-1).mse : NaN;
  const aFinalLoss = aEpochs.length ? aEpochs.at(-1).mse : NaN;
  addComparisonRow(trainingEl, "Epochs", String(pEpochs.length), String(aEpochs.length));
  addComparisonRow(
    trainingEl, "Final loss",
    Number.isFinite(pFinalLoss) ? pFinalLoss.toFixed(8) : "n/a",
    Number.isFinite(aFinalLoss) ? aFinalLoss.toFixed(8) : "n/a",
  );

  const valEl = byId("compare-validation");
  valEl.replaceChildren();
  const pm = primary.evaluation.metrics;
  const am = alt.evaluation.metrics;
  for (const [key, label] of [
    ["accuracy", "Accuracy"],
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1_score", "F1"],
  ]) {
    addComparisonRow(valEl, label, pct(pm[key]), pct(am[key]));
  }
  addComparisonRow(valEl, "Latency",
    `${pm.inference_latency_ms.toFixed(1)} ms`,
    `${am.inference_latency_ms.toFixed(1)} ms`);

  const calEl = byId("compare-calibration");
  calEl.replaceChildren();
  const pc = primary.training.calibration;
  const ac = alt.training.calibration;
  addComparisonRow(calEl, "Threshold",
    pc.threshold.toFixed(8), ac.threshold.toFixed(8));
  addComparisonRow(calEl, "Mean error",
    pc.mean_error.toFixed(8), ac.mean_error.toFixed(8));
  addComparisonRow(calEl, "Std error",
    pc.std_error.toFixed(8), ac.std_error.toFixed(8));

  const alertEl = byId("compare-alerts");
  alertEl.replaceChildren();
  const pAlerts = primary.evaluation.observations.filter(
    (o) => o.prediction === "anomaly").length;
  const aAlerts = alt.evaluation.observations.filter(
    (o) => o.prediction === "anomaly").length;
  addComparisonRow(alertEl, "Alerts", `${pAlerts} / ${primary.dataset.test_total}`,
    `${aAlerts} / ${alt.dataset.test_total}`);
  addComparisonRow(alertEl, "TP", String(pm.true_positives), String(am.true_positives));
  addComparisonRow(alertEl, "FP", String(pm.false_positives), String(am.false_positives));
  addComparisonRow(alertEl, "FN", String(pm.false_negatives), String(am.false_negatives));
  addComparisonRow(alertEl, "TN", String(pm.true_negatives), String(am.true_negatives));

  byId("comparison-note").textContent =
    `Both runs use the same synthetic dataset (${primary.dataset.test_total} test records, ` +
    `5% anomaly rate). Seed ${primary.config.seed} threshold: ${pc.threshold.toFixed(8)}; ` +
    `Seed ${alt.config.seed} threshold: ${ac.threshold.toFixed(8)}.`;

  const a11y = byId("comparison-a11y");
  if (a11y) {
    a11y.textContent =
      `Comparison loaded. Seed ${primary.config.seed}: accuracy ${pct(pm.accuracy)}, ` +
      `F1 ${pct(pm.f1_score)}, threshold ${pc.threshold.toFixed(8)}, ` +
      `${pAlerts} alerts. Seed ${alt.config.seed}: accuracy ${pct(am.accuracy)}, ` +
      `F1 ${pct(am.f1_score)}, threshold ${ac.threshold.toFixed(8)}, ` +
      `${aAlerts} alerts.`;
  }
}

function addComparisonRow(container, label, primaryVal, altVal) {
  const row = el("div", null, "comparison-row");
  row.appendChild(el("span", label, "comparison-label"));
  row.appendChild(el("span", primaryVal, "comparison-value"));
  row.appendChild(el("span", altVal, "comparison-value"));
  container.appendChild(row);
}

// --- Live simulation (fault injection) ---

const liveState = {
  enabled: false,
  type: "none",
  severity: 0.5,
  phase: "all",
};

function applyLiveSimulation() {
  if (!liveState.enabled || !state.data || !state.records.length) return;
  const record = state.records[state.current];
  if (!record) return;

  const thdLimit = state.data.config.thd_limit;
  const calibration = state.data.training.calibration;

  const modified = deepCopyRecord(record);
  applyFault(modified);

  const reconstructionRatio = evidenceRatio(modified.reconstruction_error, calibration.threshold);
  const thdRatio = evidenceRatio(modified.thd_ratio, thdLimit);
  const isAnomaly = reconstructionRatio >= 1 || thdRatio >= 1;

  const badge = byId("live-sim-badge");
  const detail = byId("live-sim-detail");
  badge.dataset.state = isAnomaly ? "anomaly" : "normal";
  badge.textContent = isAnomaly ? "ALARM" : "normal";
  detail.textContent = isAnomaly
    ? `${liveState.type.replace(/_/g, " ")}: recon ${(reconstructionRatio * 100).toFixed(0)}%, THD ${(thdRatio * 100).toFixed(0)}%`
    : `${liveState.type.replace(/_/g, " ")}: below threshold`;
}

function deepCopyRecord(record) {
  return {
    ...record,
    actual: record.actual.map((phase) => [...phase]),
    reconstruction: record.reconstruction.map((phase) => [...phase]),
    triggers: [...record.triggers],
  };
}

function applyFault(record) {
  const { type, severity, phase } = liveState;
  if (type === "none") return;

  const phases = phase === "all" ? [0, 1, 2] : [Number(phase)];
  const amplitude = severity * 0.4;

  for (const p of phases) {
    const actual = record.actual[p];
    const recon = record.reconstruction[p];
    const len = actual.length;

    switch (type) {
      case "amplitude_sag":
        for (let i = 0; i < len; i++) actual[i] *= (1 - amplitude);
        break;
      case "amplitude_swell":
        for (let i = 0; i < len; i++) actual[i] *= (1 + amplitude);
        break;
      case "phase_offset": {
        const shift = Math.floor(amplitude * len * 0.3);
        const shifted = new Array(len);
        for (let i = 0; i < len; i++) shifted[i] = actual[(i + shift) % len];
        for (let i = 0; i < len; i++) actual[i] = shifted[i];
        break;
      }
      case "harmonic_injection":
        for (let i = 0; i < len; i++) {
          const t = i / len;
          actual[i] += amplitude * Math.sin(2 * Math.PI * 5 * t);
        }
        break;
      case "noise":
        for (let i = 0; i < len; i++) {
          actual[i] += amplitude * (Math.random() * 2 - 1);
        }
        break;
      case "dc_offset":
        for (let i = 0; i < len; i++) actual[i] += amplitude * 0.5;
        break;
      case "frequency_drift": {
        const drift = amplitude * 0.08;
        for (let i = 0; i < len; i++) {
          const t = i / len;
          actual[i] *= Math.cos(2 * Math.PI * (1 + drift * t) * (len / 512));
        }
        break;
      }
    }
  }

  let totalError = 0;
  for (let p = 0; p < 3; p++) {
    let phaseError = 0;
    for (let i = 0; i < record.actual[p].length; i++) {
      const diff = record.actual[p][i] - record.reconstruction[p][i];
      phaseError += diff * diff;
    }
    totalError += phaseError / record.actual[p].length;
  }
  record.reconstruction_error = Math.sqrt(totalError / 3);

  let maxThd = 0;
  for (let p = 0; p < 3; p++) {
    const signal = record.actual[p];
    const thd = computeSimpleThd(signal);
    if (thd > maxThd) maxThd = thd;
  }
  record.thd_ratio = maxThd;
  record.prediction = (record.reconstruction_error > state.data.training.calibration.threshold || maxThd > state.data.config.thd_limit) ? "anomaly" : "normal";
  record.triggers = [];
  if (record.reconstruction_error > state.data.training.calibration.threshold) record.triggers.push("reconstruction_error");
  if (maxThd > state.data.config.thd_limit) record.triggers.push("thd");
}

function computeSimpleThd(signal) {
  const n = signal.length;
  if (n < 16) return 0;
  let fundRe = 0, fundIm = 0;
  for (let i = 0; i < n; i++) {
    const angle = 2 * Math.PI * i / n;
    fundRe += signal[i] * Math.cos(angle);
    fundIm += signal[i] * Math.sin(angle);
  }
  fundRe /= n;
  fundIm /= n;
  const fundAmp = Math.sqrt(fundRe * fundRe + fundIm * fundIm);
  if (fundAmp < 1e-12) return 1;
  let harmonicPower = 0;
  for (let h = 2; h <= 50; h++) {
    let hRe = 0, hIm = 0;
    for (let i = 0; i < n; i++) {
      const angle = 2 * Math.PI * h * i / n;
      hRe += signal[i] * Math.cos(angle);
      hIm += signal[i] * Math.sin(angle);
    }
    hRe /= n;
    hIm /= n;
    harmonicPower += hRe * hRe + hIm * hIm;
  }
  return Math.sqrt(harmonicPower) / fundAmp;
}

byId("live-sim-enabled").addEventListener("change", (event) => {
  liveState.enabled = event.target.checked;
  const controls = byId("live-sim-controls");
  controls.classList.toggle("active", liveState.enabled);
  if (!liveState.enabled) {
    byId("live-sim-badge").dataset.state = "normal";
    byId("live-sim-badge").textContent = "normal";
    byId("live-sim-detail").textContent = "No fault active";
    renderCurrentRecord();
  } else {
    applyLiveSimulation();
    drawWaveformWithFault();
  }
});

byId("fault-type").addEventListener("change", (event) => {
  liveState.type = event.target.value;
  if (liveState.enabled) {
    applyLiveSimulation();
    drawWaveformWithFault();
  }
});

byId("fault-severity").addEventListener("input", (event) => {
  liveState.severity = Number(event.target.value);
  byId("fault-severity-value").textContent = liveState.severity.toFixed(2);
  if (liveState.enabled) {
    applyLiveSimulation();
    drawWaveformWithFault();
  }
});

byId("fault-phase").addEventListener("change", (event) => {
  liveState.phase = event.target.value;
  if (liveState.enabled) {
    applyLiveSimulation();
    drawWaveformWithFault();
  }
});

function drawWaveformWithFault() {
  if (!liveState.enabled || !state.data || !state.records.length) return;
  const record = state.records[state.current];
  if (!record) return;

  const modified = deepCopyRecord(record);
  applyFault(modified);

  const canvas = byId("waveform-canvas");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const margin = { left: 64, right: 24, top: 24, bottom: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const colors = ["#36d6c7", "#7aa8ff", "#ffb84d"];
  const pointCount = state.data.replay.time_ms.length;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#081014";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "#23363a";
  context.fillStyle = "#9fb6b0";
  context.font = "14px ui-monospace, monospace";
  context.lineWidth = 1;
  for (let index = 0; index <= 6; index += 1) {
    const x = margin.left + (plotWidth * index) / 6;
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, margin.top + plotHeight);
    context.stroke();
    const time = state.data.replay.time_ms[Math.round(((pointCount - 1) * index) / 6)];
    context.fillText(`${fixed(time, 1)}`, x - 14, height - 18);
  }
  for (let index = 0; index <= 4; index += 1) {
    const y = margin.top + (plotHeight * index) / 4;
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();
    context.fillText((1.5 - index * 0.75).toFixed(2), 10, y + 5);
  }

  const yFor = (value) => margin.top + plotHeight / 2 - (finiteNumber(value) / 1.5) * (plotHeight / 2);
  for (let phase = 0; phase < 3; phase += 1) {
    const actual = modified.actual[phase];
    const reconstruction = record.reconstruction[phase];
    context.strokeStyle = colors[phase];
    context.lineWidth = 2.5;
    context.setLineDash([]);
    context.beginPath();
    actual.forEach((value, index) => {
      const x = margin.left + (plotWidth * index) / Math.max(actual.length - 1, 1);
      const y = yFor(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();

    context.strokeStyle = "#9fb6b0";
    context.lineWidth = 1.2;
    context.setLineDash([7, 7]);
    context.beginPath();
    reconstruction.forEach((value, index) => {
      const x = margin.left + (plotWidth * index) / Math.max(reconstruction.length - 1, 1);
      const y = yFor(value);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  }
  context.setLineDash([]);
  const cursorX = margin.left + plotWidth * state.scan;
  context.strokeStyle = "#eef7f3";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(cursorX, margin.top);
  context.lineTo(cursorX, margin.top + plotHeight);
  context.stroke();
}

// --- Run selector ---

byId("run-select").addEventListener("change", (event) => {
  const url = event.target.value;
  const label = event.target.options[event.target.selectedIndex].text;
  state.activeRunUrl = url;
  state.comparisonData = null;
  byId("comparison-section").hidden = true;
  const status = byId("run-select-status");
  if (status) status.textContent = `Loading ${label}...`;
  loadResults({ announce: true, url }).then(() => {
    if (status) status.textContent = `Loaded ${label}`;
  });
});

// --- DSP Engine Integration ---

let dspFrameCount = 0;
let dspLastFpsTime = performance.now();

function updateDspEngine() {
  if (!state.data || !state.records.length) return;
  const record = state.records[state.current];
  if (!record) return;

  // Use the actual waveform data from the record
  const va = record.actual[0];
  const vb = record.actual[1];
  const vc = record.actual[2];

  // Apply fault if live simulation is active
  let vaFault = va, vbFault = vb, vcFault = vc;
  if (liveState.enabled) {
    const modified = deepCopyRecord(record);
    applyFault(modified);
    vaFault = modified.actual[0];
    vbFault = modified.actual[1];
    vcFault = modified.actual[2];
  }

  // Compute Clarke Transform
  const clarke = clarkeTransformBatch(vaFault, vbFault, vcFault);

  // Render phase portrait
  const canvas = byId("phase-portrait-canvas");
  if (canvas) {
    renderPhasePortrait(canvas, clarke, {
      trailLength: 2048,
      isFault: liveState.enabled && liveState.type !== "none",
    });
  }

  // Run autoencoder analysis
  const analysis = forwardPass(vaFault, vbFault, vcFault);

  // Update DSP metrics
  setText("clarke-alpha", clarke.alpha[clarke.alpha.length - 1]?.toFixed(4) || "0.0000");
  setText("clarke-beta", clarke.beta[clarke.beta.length - 1]?.toFixed(4) || "0.0000");
  setText("clarke-zero", clarke.zero[clarke.zero.length - 1]?.toFixed(4) || "0.0000");

  setText("dsp-imbalance", `${(analysis.imbalance * 100).toFixed(2)}%`);
  setText("dsp-zero-seq", `${(analysis.zeroSequence * 100).toFixed(2)}%`);
  setText("dsp-harmonic-ratio", `${(analysis.harmonicRatio * 100).toFixed(2)}%`);

  const domH = analysis.dominantHarmonic;
  setText("dsp-dominant-h", domH.harmonic > 0 ? `H${domH.harmonic}` : "—");

  // Update progress meters
  const imbalanceMeter = byId("dsp-imbalance-meter");
  const zeroSeqMeter = byId("dsp-zero-seq-meter");
  const harmonicMeter = byId("dsp-harmonic-ratio-meter");
  if (imbalanceMeter) imbalanceMeter.value = Math.min(analysis.imbalance * 5, 1);
  if (zeroSeqMeter) zeroSeqMeter.value = Math.min(analysis.zeroSequence * 5, 1);
  if (harmonicMeter) harmonicMeter.value = Math.min(analysis.harmonicRatio * 5, 1);

  // Update decision badge
  const badge = byId("dsp-decision-badge");
  if (badge) {
    badge.dataset.state = analysis.prediction === "anomaly" ? "anomaly" : "normal";
    badge.textContent = analysis.prediction === "anomaly" ? "ALARM" : "normal";
  }

  // Update FPS counter
  dspFrameCount++;
  const now = performance.now();
  if (now - dspLastFpsTime >= 1000) {
    setText("dsp-fps", `${dspFrameCount} FPS`);
    dspFrameCount = 0;
    dspLastFpsTime = now;
  }

  // Render harmonic spectrum bar chart
  renderHarmonicSpectrum(vaFault, vbFault, vcFault, analysis);
}

function renderHarmonicSpectrum(va, vb, vc, analysis) {
  const canvas = byId("harmonic-spectrum-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const n = va.length;
  if (n < 16) return;

  // FFT using DFT for harmonic orders 1-50
  const sampleRate = 10000;
  const fundamental = 60;
  const maxOrder = 50;
  const magnitudes = new Float64Array(maxOrder + 1);

  // Use Phase A for spectrum (representative)
  const signal = va;
  const mean = signal.reduce((s, v) => s + v, 0) / n;
  const centered = signal.map(v => v - mean);

  for (let order = 1; order <= maxOrder; order++) {
    const freq = fundamental * order;
    const omega = (2 * Math.PI * freq) / sampleRate;
    let re = 0, im = 0;
    for (let k = 0; k < n; k++) {
      re += centered[k] * Math.cos(omega * k);
      im -= centered[k] * Math.sin(omega * k);
    }
    magnitudes[order] = Math.sqrt(re * re + im * im) / (n / 2);
  }

  // Render bar chart — responsive: clear stale inline width before measuring
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  const W = canvas.clientWidth || 760;
  const H = Math.round(W * 280 / 760);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, W, H);

  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const plotW = W - padding.left - padding.right;
  const plotH = H - padding.top - padding.bottom;

  // Find max magnitude for scaling
  let maxMag = 0;
  for (let i = 1; i <= maxOrder; i++) {
    if (magnitudes[i] > maxMag) maxMag = magnitudes[i];
  }
  if (maxMag < 1e-12) maxMag = 1;

  // Draw bars
  const barWidth = plotW / (maxOrder + 1);
  const barGap = barWidth * 0.15;

  for (let order = 1; order <= maxOrder; order++) {
    const barH = (magnitudes[order] / maxMag) * plotH;
    const x = padding.left + (order - 0.5) * barWidth;
    const y = padding.top + plotH - barH;

    // Color: fundamental (order 1) is accent, harmonics colored by severity
    if (order === 1) {
      ctx.fillStyle = "var(--accent, #4fc3f7)";
    } else if (magnitudes[order] / maxMag > 0.1) {
      ctx.fillStyle = "var(--alarm, #ef5350)";
    } else if (magnitudes[order] / maxMag > 0.03) {
      ctx.fillStyle = "var(--warning, #ffa726)";
    } else {
      ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
    }

    ctx.fillRect(x + barGap / 2, y, barWidth - barGap, barH);

    // Order labels for every 5th harmonic
    if (order % 5 === 0 || order === 1) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(order, x + barWidth / 2, padding.top + plotH + 14);
    }
  }

  // Y-axis label
  ctx.save();
  ctx.translate(14, padding.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Magnitude (p.u.)", 0, 0);
  ctx.restore();

  // X-axis label
  ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
  ctx.font = "11px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Harmonic order", padding.left + plotW / 2, H - 4);

  // IEEE 519 expected harmonic envelope (dashed line)
  // For h < 11: 4.0%; 11 <= h < 17: 2.0%; 17 <= h < 23: 1.5%; 23 <= h < 35: 0.6%; h >= 35: 0.3%
  const ieeeLimit = (order) => {
    if (order < 11) return 0.04;
    if (order < 17) return 0.02;
    if (order < 23) return 0.015;
    if (order < 35) return 0.006;
    return 0.003;
  };
  ctx.strokeStyle = "rgba(255, 193, 7, 0.5)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (let order = 2; order <= maxOrder; order++) {
    const limitH = (ieeeLimit(order) * magnitudes[1] / maxMag) * plotH;
    const x = padding.left + (order - 0.5) * barWidth;
    const y = padding.top + plotH - limitH;
    if (order === 2) ctx.moveTo(x + barWidth / 2, y);
    else ctx.lineTo(x + barWidth / 2, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);

  // Legend for IEEE 519 envelope
  ctx.fillStyle = "rgba(255, 193, 7, 0.7)";
  ctx.font = "9px monospace";
  ctx.textAlign = "left";
  ctx.fillText("--- IEEE 519 limit", padding.left + 5, padding.top + 12);

  // Update THD readout
  const fundamentalPower = magnitudes[1] * magnitudes[1];
  let harmonicPower = 0;
  for (let i = 2; i <= maxOrder; i++) {
    harmonicPower += magnitudes[i] * magnitudes[i];
  }
  const thd = fundamentalPower > 1e-12 ? Math.sqrt(harmonicPower / fundamentalPower) : 0;
  const thdPct = (thd * 100).toFixed(2);
  const thdColor = thd < 0.05 ? "#66bb6a" : thd < 0.1 ? "#ffa726" : "#ef5350";
  setText("spectrum-thd", `THD: ${thdPct}%`);
  const thdEl = byId("spectrum-thd");
  if (thdEl) thdEl.style.color = thdColor;

  // Screen reader summary
  const dominantH = analysis?.dominantHarmonic;
  const srText = byId("harmonic-spectrum-a11y");
  if (srText) {
    srText.textContent = dominantH?.harmonic > 0
      ? `Harmonic spectrum shows dominant harmonic at order ${dominantH.harmonic}, THD ${thdPct}%. IEEE 519 limit overlay included.`
      : `Harmonic spectrum shows no dominant harmonic, THD ${thdPct}%. IEEE 519 limit overlay included.`;
  }
}

// --- Hardware Deployment ---

let generatedCppCode = "";
let generatedFilename = "";

byId("hw-generate").addEventListener("click", () => {
  const platform = byId("hw-platform").value;
  const info = getPlatformInfo(platform);

  // Update platform info
  byId("hw-platform-name").textContent = info.name;
  byId("hw-arch").textContent = info.architecture;
  byId("hw-flash").textContent = info.flash;
  byId("hw-ram").textContent = info.ram;
  byId("hw-clock").textContent = info.clock;
  byId("hw-features").textContent = info.features.join(", ");
  byId("hardware-info").hidden = false;

  // Generate code
  generatedCppCode = generateCppCode(platform);
  const extensions = { stm32: ".cpp", arduino: ".ino", esp32: ".cpp" };
  generatedFilename = `grid_sentinel_inference${extensions[platform] || ".cpp"}`;

  // Display code
  byId("hw-code-filename").textContent = generatedFilename;
  byId("hw-code-content").textContent = generatedCppCode;
  byId("hardware-code-container").hidden = false;
  byId("hw-download").disabled = false;
});

byId("hw-download").addEventListener("click", () => {
  if (generatedCppCode) {
    downloadCppFile(generatedCppCode, generatedFilename);
  }
});

byId("hw-copy").addEventListener("click", () => {
  if (generatedCppCode) {
    navigator.clipboard.writeText(generatedCppCode).then(() => {
      const btn = byId("hw-copy");
      const originalText = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = originalText; }, 2000);
    });
  }
});

updatePlayButton();
renderSimClock();
loadResults();
// Start animation loop for DSP engine updates (always running)
requestAnimationLoop();

// --- Energy Balance Panel ---
let energyHistory = [];

function updateEnergyBalance() {
  const records = state.records;
  if (!records || records.length === 0) return;

  // Estimate generation from actual voltage magnitudes (average across phases)
  let totalGen = 0;
  let totalLoad = 0;
  const recordCount = records.length;

  for (let i = 0; i < recordCount; i++) {
    const r = records[i];
    const actual = r.actual;
    if (!actual || actual.length < 3) continue;

    // Approximate generation as average RMS voltage (normalized)
    const rmsPhases = actual.map(p => Math.sqrt(p.reduce((s, v) => s + v * v, 0) / p.length));
    const avgRms = rmsPhases.reduce((s, v) => s + v, 0) / rmsPhases.length;

    // Approximate load as demand factor (inversely proportional to voltage sag)
    const demandFactor = avgRms > 0 ? 1.0 / avgRms : 1.0;

    totalGen += avgRms;
    totalLoad += demandFactor;
  }

  // Normalize to reasonable MW scale (assumes 380 records = 380 MW capacity)
  const scale = 1.0;
  const genMW = (totalGen / recordCount) * scale * 100;
  const loadMW = (totalLoad / recordCount) * scale * 80;
  const storedMWh = Math.abs(genMW - loadMW) * 2.5;
  const netBalance = genMW - loadMW;

  // Update DOM
  setText("energy-gen-value", genMW.toFixed(2));
  setText("energy-load-value", loadMW.toFixed(2));
  setText("energy-storage-value", storedMWh.toFixed(2));
  setText("energy-balance-value", netBalance.toFixed(2));

  // Color the net balance indicator
  const balanceCard = document.querySelector(".energy-balance-indicator");
  if (balanceCard) {
    balanceCard.style.borderColor = netBalance >= 0
      ? "var(--accent, #4fc3f7)"
      : "var(--alarm, #ef5350)";
  }

  // Track history for line chart (keep last 60 points)
  energyHistory.push({ gen: genMW, load: loadMW, storage: storedMWh, net: netBalance });
  if (energyHistory.length > 60) energyHistory.shift();

  // Render line chart
  renderEnergyChart();

  // Screen reader summary
  const srText = byId("energy-balance-a11y");
  if (srText) {
    srText.textContent = netBalance >= 0
      ? `Generation ${genMW.toFixed(1)} MW exceeds load ${loadMW.toFixed(1)} MW. Net surplus ${netBalance.toFixed(1)} MW, ${storedMWh.toFixed(1)} MWh stored.`
      : `Load ${loadMW.toFixed(1)} MW exceeds generation ${genMW.toFixed(1)} MW. Net deficit ${Math.abs(netBalance).toFixed(1)} MW, ${storedMWh.toFixed(1)} MWh stored.`;
  }
}

function renderEnergyChart() {
  const canvas = byId("energy-balance-canvas");
  if (!canvas || energyHistory.length < 2) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const W = canvas.width;
  const H = canvas.height;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, W, H);

  const padding = { top: 15, right: 15, bottom: 25, left: 45 };
  const plotW = W - padding.left - padding.right;
  const plotH = H - padding.top - padding.bottom;

  // Find value ranges
  let minVal = Infinity, maxVal = -Infinity;
  for (const pt of energyHistory) {
    const vals = [pt.gen, pt.load, pt.storage];
    for (const v of vals) {
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
  }
  const range = maxVal - minVal || 1;
  const margin = range * 0.1;
  minVal -= margin;
  maxVal += margin;

  const toX = (i) => padding.left + (i / (energyHistory.length - 1)) * plotW;
  const toY = (v) => padding.top + plotH - ((v - minVal) / (maxVal - minVal)) * plotH;

  // Draw grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i < 5; i++) {
    const y = padding.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + plotW, y);
    ctx.stroke();
  }

  // Draw lines
  const lines = [
    { key: "gen", color: "#4fc3f7", label: "Gen" },
    { key: "load", color: "#ef5350", label: "Load" },
    { key: "storage", color: "#66bb6a", label: "Stored" },
  ];

  for (const line of lines) {
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < energyHistory.length; i++) {
      const x = toX(i);
      const y = toY(energyHistory[i][line.key]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Legend
  ctx.font = "10px monospace";
  let lx = padding.left + 5;
  for (const line of lines) {
    ctx.fillStyle = line.color;
    ctx.fillRect(lx, H - 15, 12, 3);
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.textAlign = "left";
    ctx.fillText(line.label, lx + 16, H - 11);
    lx += 70;
  }
}

// --- Grid Health Indicators (DQ Balance, PLL Quality, MPPT, Reliability) ---

function renderGridHealth(data) {
  const observations = data.evaluation?.observations;
  if (!observations || !observations.length) return;

  // Compute DQ balance indicator from reconstruction errors and THD
  // High THD with low reconstruction error suggests DQ-domain imbalance
  const thdValues = observations.map((o) => o.thd_ratio).filter((v) => v != null && isFinite(v));
  const reconErrors = observations.map((o) => o.reconstruction_error).filter((v) => v != null && isFinite(v));
  const avgThd = thdValues.length ? thdValues.reduce((a, b) => a + b, 0) / thdValues.length : 0;
  const maxThd = thdValues.length ? Math.max(...thdValues) : 0;

  // DQ balance: ratio of harmonic power to fundamental (lower is better)
  // Normal: < 0.05 (5%); Warning: 0.05-0.15; Alarm: > 0.15
  const dqBalance = Math.min(avgThd * 10, 1.0);
  const dqPct = Math.round(dqBalance * 100);
  const dqCard = byId("dq-balance-card");
  const dqValue = byId("dq-balance-value");
  const dqBar = byId("dq-balance-bar");
  const dqStatus = byId("dq-balance-status");
  if (dqValue) dqValue.textContent = dqPct + "%";
  if (dqBar) dqBar.style.width = dqPct + "%";
  if (dqCard) {
    if (avgThd < 0.05) { dqCard.dataset.state = "nominal"; if (dqStatus) dqStatus.textContent = "NOMINAL"; }
    else if (avgThd < 0.15) { dqCard.dataset.state = "warning"; if (dqStatus) dqStatus.textContent = "WARNING"; }
    else { dqCard.dataset.state = "alarm"; if (dqStatus) dqStatus.textContent = "ALARM"; }
  }

  // PLL phase error: derived from reconstruction error distribution
  // Normal: < 2 degrees; Warning: 2-5; Alarm: > 5
  const avgRecon = reconErrors.length ? reconErrors.reduce((a, b) => a + b, 0) / reconErrors.length : 0;
  const pllError = Math.min(avgRecon * 50, 10); // scale to degrees
  const pllCard = byId("pll-quality-card");
  const pllValue = byId("pll-error-value");
  const pllBar = byId("pll-error-bar");
  const pllStatus = byId("pll-error-status");
  if (pllValue) pllValue.textContent = pllError.toFixed(1) + "\u00B0";
  if (pllBar) pllBar.style.width = Math.min(pllError / 10 * 100, 100) + "%";
  if (pllCard) {
    if (pllError < 2) { pllCard.dataset.state = "nominal"; if (pllStatus) pllStatus.textContent = "NOMINAL"; }
    else if (pllError < 5) { pllCard.dataset.state = "warning"; if (pllStatus) pllStatus.textContent = "WARNING"; }
    else { pllCard.dataset.state = "alarm"; if (pllStatus) pllStatus.textContent = "ALARM"; }
  }

  // MPPT efficiency: inverse of max THD (cleaner signal = better tracking)
  // Normal: > 95%; Warning: 90-95%; Alarm: < 90%
  const mpptEff = Math.max(0, Math.min(100, 100 - maxThd * 200));
  const mpptCard = byId("mppt-efficiency-card");
  const mpptValue = byId("mppt-eff-value");
  const mpptBar = byId("mppt-eff-bar");
  const mpptStatus = byId("mppt-eff-status");
  if (mpptValue) mpptValue.textContent = mpptEff.toFixed(1) + "%";
  if (mpptBar) mpptBar.style.width = mpptEff + "%";
  if (mpptCard) {
    if (mpptEff > 95) { mpptCard.dataset.state = "nominal"; if (mpptStatus) mpptStatus.textContent = "NOMINAL"; }
    else if (mpptEff > 90) { mpptCard.dataset.state = "warning"; if (mpptStatus) mpptStatus.textContent = "WARNING"; }
    else { mpptCard.dataset.state = "alarm"; if (mpptStatus) mpptStatus.textContent = "ALARM"; }
  }

  // Reliability index from grid_response (SAIFI)
  const gr = data.grid_response;
  const relCard = byId("reliability-card");
  const relValue = byId("reliability-saifi-value");
  const relBar = byId("reliability-bar");
  const relStatus = byId("reliability-status");
  if (gr && gr.reliability_indices) {
    const saifi = gr.reliability_indices.SAIFI;
    const saidi = gr.reliability_indices.SAIDI;
    if (relValue) relValue.textContent = saifi.toFixed(4);
    if (relBar) relBar.style.width = Math.min(saifi * 5000, 100) + "%";
    if (relCard) {
      if (saifi < 0.001) { relCard.dataset.state = "nominal"; if (relStatus) relStatus.textContent = "NOMINAL"; }
      else if (saifi < 0.005) { relCard.dataset.state = "warning"; if (relStatus) relStatus.textContent = "WARNING"; }
      else { relCard.dataset.state = "alarm"; if (relStatus) relStatus.textContent = "ALARM"; }
    }
  }

  // Screen reader summary
  const srText = byId("grid-health-a11y");
  if (srText) {
    srText.textContent = `Grid health: DQ balance ${dqPct}%, PLL phase error ${pllError.toFixed(1)} degrees, ` +
      `MPPT efficiency ${mpptEff.toFixed(1)}%` +
      (gr && gr.reliability_indices ? `, SAIFI ${gr.reliability_indices.SAIFI.toFixed(4)}` : "") + ".";
  }
}

// --- 3D Topology Bidirectional Interaction ---
document.addEventListener("grid-sentinel:section-click", (event) => {
  const { section } = event.detail;
  // Find the first record belonging to this section
  const targetRecord = state.records.findIndex(
    (r) => sectionOf(r.source_index) === section,
  );
  if (targetRecord >= 0) {
    selectRecord(targetRecord);
  }
  // Auto-enable live simulation with a default fault type
  if (!liveState.enabled) {
    liveState.enabled = true;
    liveState.type = "amplitude_sag";
    liveState.severity = 0.5;
    byId("live-sim-enabled").checked = true;
    byId("live-sim-controls").classList.add("active");
    byId("fault-type").value = "amplitude_sag";
    byId("fault-severity").value = "0.5";
    byId("fault-severity-value").textContent = "0.50";
  }
  applyLiveSimulation();
  drawWaveformWithFault();

  // Scroll to the topology panel for visibility
  const panel = document.querySelector(".topology-panel");
  if (panel) panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
});
