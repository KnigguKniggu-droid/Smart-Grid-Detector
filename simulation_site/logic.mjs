// Pure, DOM-free logic shared by the dashboard and its tests. The browser
// imports it as an ES module; tests import the same functions directly.
// Keeping one source of truth means the tested behavior is the shipped
// behavior.

export const SECTIONS = 8;
export const MAX_REPLAY_RECORDS = 64;
export const MAX_REPLAY_POINTS = 4096;
export const MAX_OBSERVATIONS = 50_000;
export const MAX_TRAINING_EPOCHS = 500;

// Fixed-time-stepping simulation clock, after the Omniverse Kit timeline model:
// a wall-clock accumulator converts variable frame times into whole ticks of
// 1/timeCodesPerSecond seconds, with a catch-up clamp so a long stall cannot
// fast-forward the whole replay at once.
export function createSimClock(overrides = {}) {
  return {
    timeCodesPerSecond: 60,
    ticksPerSweep: 250,
    maxCatchUpTicks: 8,
    accumulator: 0,
    tick: 0,
    droppedTicks: 0,
    everyFrame: false,
    ...overrides,
    advance(wallDeltaSeconds, speed) {
      if (this.everyFrame) {
        this.tick += 1;
        return 1;
      }
      this.accumulator += wallDeltaSeconds * speed;
      const dt = 1 / this.timeCodesPerSecond;
      let ticks = Math.floor(this.accumulator / dt);
      this.accumulator -= ticks * dt;
      if (ticks > this.maxCatchUpTicks) {
        this.droppedTicks += ticks - this.maxCatchUpTicks;
        ticks = this.maxCatchUpTicks;
      }
      this.tick += ticks;
      return ticks;
    },
  };
}

// Adaptive polling: snap back to the base cadence when a new run lands,
// otherwise back off geometrically up to the ceiling.
export function nextPollDelay(current, changed, base, max, backoff) {
  return changed ? base : Math.min(current * backoff, max);
}

export function sectionOf(sourceIndex, sections = SECTIONS) {
  return ((sourceIndex % sections) + sections) % sections;
}

// Per-section alert aggregation used by the 3D map and its accessibility
// summary: records, alerts, and peak THD for each feeder section.
export function aggregateSections(observations, sections = SECTIONS) {
  const stats = Array.from({ length: sections }, () => ({
    records: 0,
    alerts: 0,
    maxThd: 0,
  }));
  for (const item of observations || []) {
    const bucket = stats[sectionOf(item.source_index, sections)];
    bucket.records += 1;
    if (item.prediction === "anomaly") bucket.alerts += 1;
    if (Number.isFinite(item.thd_ratio)) {
      bucket.maxThd = Math.max(bucket.maxThd, item.thd_ratio);
    }
  }
  return stats;
}

function isEvidence(value) {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function assertDecisionRecord(record, label) {
  const sampleIndex = record?.sample_index ?? record?.index;
  if (!record || !Number.isInteger(sampleIndex) || sampleIndex < 0) {
    throw new Error(`${label} has an invalid sample index.`);
  }
  if (!Number.isInteger(record.source_index) || record.source_index < 0) {
    throw new Error(`${label} has an invalid source index.`);
  }
  if (!['normal', 'anomaly'].includes(record.prediction)) {
    throw new Error(`${label} has an invalid prediction.`);
  }
  if (!isEvidence(record.reconstruction_error) || !isEvidence(record.thd_ratio)) {
    throw new Error(`${label} has invalid detector evidence.`);
  }
  if (!Array.isArray(record.triggers) || record.triggers.length > 3 ||
      record.triggers.some((item) => typeof item !== 'string')) {
    throw new Error(`${label} has invalid trigger evidence.`);
  }
}

export function assertResultsShape(data) {
  if (!data || data.schema_version !== 2) {
    throw new Error("Unsupported or missing results schema. Run --export-all again.");
  }
  if (!data.run || data.run.status !== "complete" || typeof data.run.id !== "string") {
    throw new Error("The latest simulation run is not complete.");
  }
  if (!data.config || !Number.isFinite(data.config.thd_limit) ||
      data.config.thd_limit <= 0) {
    throw new Error("The detector configuration is invalid.");
  }
  if (!data.training || !Array.isArray(data.training.epochs) ||
      data.training.epochs.length < 1 ||
      data.training.epochs.length > MAX_TRAINING_EPOCHS ||
      data.training.epochs.some((item, index) =>
        !item || item.epoch !== index + 1 ||
        !Number.isFinite(item.mse) || item.mse < 0 ||
        !Number.isFinite(item.latency_ms) || item.latency_ms < 0) ||
      !Number.isFinite(data.training?.calibration?.threshold) ||
      data.training.calibration.threshold <= 0) {
    throw new Error("Training or calibration evidence is invalid.");
  }
  if (!data.model || !Number.isInteger(data.model.parameters) || data.model.parameters < 1) {
    throw new Error("Model metadata is invalid.");
  }
  const observations = data.evaluation?.observations;
  if (!Array.isArray(observations) || observations.length < 1 ||
      observations.length > MAX_OBSERVATIONS) {
    throw new Error("Evaluation observations are missing or exceed the dashboard limit.");
  }
  observations.forEach((record, index) =>
    assertDecisionRecord(record, `Observation ${index}`));

  const records = data.replay?.records;
  const time = data.replay?.time_ms;
  if (!Array.isArray(records) || records.length < 1 ||
      records.length > MAX_REPLAY_RECORDS) {
    throw new Error("Recorded waveform replay count is invalid.");
  }
  if (!Array.isArray(time) || time.length < 2 || time.length > MAX_REPLAY_POINTS ||
      time.some((value) => !Number.isFinite(value))) {
    throw new Error("Replay time axis is invalid.");
  }
  for (const [recordIndex, record] of records.entries()) {
    assertDecisionRecord(record, `Replay record ${recordIndex}`);
    for (const field of ["actual", "reconstruction"]) {
      if (!Array.isArray(record[field]) || record[field].length !== 3) {
        throw new Error(`Replay record ${recordIndex} does not contain three ${field} phases.`);
      }
      for (const phase of record[field]) {
        if (!Array.isArray(phase) || phase.length !== time.length ||
            phase.some((value) => !Number.isFinite(value))) {
          throw new Error(`Replay record ${recordIndex} has invalid ${field} samples.`);
        }
      }
    }
  }
  for (const curve of Object.values(data.roc_analysis || {})) {
    if (!curve || !Array.isArray(curve.fpr)) continue;
    if (!Array.isArray(curve.tpr) || curve.fpr.length !== curve.tpr.length ||
        curve.fpr.length < 2 || curve.fpr.length > 10_000 ||
        [...curve.fpr, ...curve.tpr].some(
          (value) => !Number.isFinite(value) || value < 0 || value > 1)) {
      throw new Error("ROC evidence is malformed.");
    }
  }
  return data;
}

export function assertDispatchShape(artifact, runId, resultsSha256) {
  if (!artifact || artifact.schema_version !== 1 || artifact.run_id !== runId) {
    throw new Error("Dispatch artifact does not belong to the displayed run.");
  }
  if (!/^[a-f0-9]{64}$/.test(artifact.results_sha256 || "") ||
      artifact.results_sha256 !== resultsSha256) {
    throw new Error("Dispatch artifact is not bound to the displayed results.");
  }
  if (!Array.isArray(artifact.detected_assets) ||
      !Array.isArray(artifact.operational_dispatches) ||
      artifact.detected_assets.length > MAX_OBSERVATIONS ||
      artifact.operational_dispatches.length > MAX_OBSERVATIONS) {
    throw new Error("Dispatch artifact arrays are invalid.");
  }
  for (const [index, dispatch] of artifact.operational_dispatches.entries()) {
    if (!dispatch || typeof dispatch.dispatch_id !== "string" ||
        typeof dispatch.target_node !== "string" ||
        typeof dispatch.action_required !== "string" ||
        !["CRITICAL", "HIGH", "MEDIUM"].includes(dispatch.priority)) {
      throw new Error(`Dispatch ${index} is malformed.`);
    }
  }
  return artifact;
}

export function dispatchCacheMatches(
  cachedRunId, cachedResultsSha256, runId, resultsSha256,
) {
  return Boolean(
    cachedRunId && cachedResultsSha256 &&
    cachedRunId === runId && cachedResultsSha256 === resultsSha256,
  );
}

const api = {
  SECTIONS,
  createSimClock,
  nextPollDelay,
  sectionOf,
  aggregateSections,
  assertResultsShape,
  assertDispatchShape,
  dispatchCacheMatches,
};
export default api;
