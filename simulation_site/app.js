import {
  assertDispatchShape,
  assertResultsShape,
  createSimClock,
  dispatchCacheMatches,
  nextPollDelay,
} from "./logic.mjs";

const MAX_RESULTS_BYTES = 5 * 1024 * 1024;
const FETCH_TIMEOUT_MS = 10_000;
const RESULTS_URL = "simulation_results.json";
// Adaptive polling: conditional requests back off while the recorded artifact
// is unchanged and snap back to the base cadence when a new run lands.
const POLL_BASE_MS = 5_000;
const POLL_MAX_MS = 60_000;
const POLL_BACKOFF = 1.5;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

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
  resultsSha256: null,
};

const DISPATCHES_URL = "grid_dispatches.json";

// Fixed-time-stepping simulation clock, imported from the DOM-free module that
// the Node suite exercises directly.
const simClock = createSimClock();
window.GridReplay = {
  get state() { return state; },
  get tick() { return simClock.tick; },
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

async function loadResults({ announce = true } = {}) {
  if (state.loading) return;
  if (document.hidden) {
    schedulePoll();
    return;
  }
  state.loading = true;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  const refreshButton = byId("refresh-button");
  refreshButton.disabled = true;
  if (announce) setText("refresh-status", "Loading recorded run…");

  let timedOut = false;
  const timeoutHandle = setTimeout(() => {
    timedOut = true;
    state.controller.abort();
  }, FETCH_TIMEOUT_MS);
  try {
    const headers = {};
    if (state.validators.etag) headers["If-None-Match"] = state.validators.etag;
    if (state.validators.lastModified) {
      headers["If-Modified-Since"] = state.validators.lastModified;
    }
    const response = await fetch(RESULTS_URL, {
      cache: "no-store",
      credentials: "same-origin",
      headers,
      signal: state.controller.signal,
    });
    if (response.status === 304 && state.data) {
      state.pollDelay = nextPollDelay(
        state.pollDelay, false, POLL_BASE_MS, POLL_MAX_MS, POLL_BACKOFF,
      );
      setText(
        "refresh-status",
        `Unchanged · checked ${new Date().toLocaleTimeString()} · ${pollStatusSuffix()}`,
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
    state.validators.etag = response.headers.get("etag");
    state.validators.lastModified = response.headers.get("last-modified");
    const changedRun = data.run.id !== state.data?.run?.id;
    state.pollDelay = nextPollDelay(
      state.pollDelay, changedRun, POLL_BASE_MS, POLL_MAX_MS, POLL_BACKOFF,
    );
    state.data = data;
    state.resultsSha256 = resultsSha256;
    state.records = data.replay.records;
    if (changedRun) {
      state.current = 0;
      state.scan = 0;
      simClock.tick = 0;
      simClock.droppedTicks = 0;
    }
    renderAll();
    byId("state-light").className = "state-light is-ready";
    setText("run-status", "Recorded run ready");
    setText(
      "refresh-status",
      `Data current · ${new Date().toLocaleTimeString()} · ${pollStatusSuffix()}`,
    );
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
    schedulePoll();
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
    const response = await fetch(DISPATCHES_URL, {
      cache: "no-store",
      credentials: "same-origin",
      signal: controller.signal,
    });
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
    state.dispatchRunId = runId;
    state.dispatchResultsSha256 = resultsSha256;
  } catch (error) {
    if (requestId === state.dispatchRequestId) {
      state.dispatches = null; // Panel simply stays hidden if unavailable.
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
    ].some((id) => !byId(id).hidden);
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
    state.dispatchRunId = null;
    state.dispatchResultsSha256 = null;
  }
  renderRobustnessPanel(data.multi_seed);
  renderThresholdPanel(data.threshold_sweep);
  renderRocPanel(data.roc_analysis);
  renderResiliencePanel(data);
  renderDispatchPanel();
  const anyVisible = [
    "robustness-panel",
    "threshold-panel",
    "roc-panel",
    "resilience-panel",
    "dispatch-panel",
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
  drawWaveform(record);
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
  if (state.playing) requestAnimationLoop();
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

updatePlayButton();
renderSimClock();
loadResults();
if (state.playing) requestAnimationLoop();
