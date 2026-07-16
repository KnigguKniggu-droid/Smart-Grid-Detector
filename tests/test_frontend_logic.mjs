// Frontend logic tests. Run with: node --test tests/test_frontend_logic.mjs
// Exercises the same DOM-free module the dashboard ships (simulation_site/
// logic.mjs), so the tested behavior is the shipped behavior.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  createSimClock,
  nextPollDelay,
  sectionOf,
  aggregateSections,
  assertResultsShape,
  assertDispatchShape,
  dispatchCacheMatches,
  classifySaifi,
  classifySaidi,
  formatReliabilityRow,
  classifyMpptEfficiency,
  classifyMismatch,
  mpptSectionStatus,
  classifyRipple,
  classifyPllError,
  pllLockStatus,
  extractReliabilityIndices,
  extractMpptMetrics,
  extractDqPllMetrics,
} from "../simulation_site/logic.mjs";

test("simClock accumulates whole ticks at the configured rate", () => {
  const clock = createSimClock();
  // 60 ticks/sec, dt = 1/60 s. 0.05 s of wall time is exactly 3 ticks.
  assert.equal(clock.advance(0.05, 1), 3);
  assert.equal(clock.tick, 3);
  // Speed 2 doubles the advance for the same wall time.
  assert.equal(clock.advance(0.05, 2), 6);
  assert.equal(clock.tick, 9);
});

test("simClock clamps catch-up and records dropped ticks", () => {
  const clock = createSimClock();
  // 1 s of wall time is 60 ticks; only maxCatchUpTicks (8) are applied.
  assert.equal(clock.advance(1.0, 1), 8);
  assert.equal(clock.tick, 8);
  assert.equal(clock.droppedTicks, 52);
});

test("simClock every-frame mode advances exactly one tick per call", () => {
  const clock = createSimClock({ everyFrame: true });
  assert.equal(clock.advance(999, 1), 1);
  assert.equal(clock.advance(0, 1), 1);
  assert.equal(clock.tick, 2);
  assert.equal(clock.droppedTicks, 0);
});

test("nextPollDelay resets on change and backs off to the ceiling otherwise", () => {
  const base = 5000;
  const max = 60000;
  const backoff = 1.5;
  assert.equal(nextPollDelay(30000, true, base, max, backoff), base);
  assert.equal(nextPollDelay(base, false, base, max, backoff), 7500);
  // Repeated backoff never exceeds the ceiling.
  let delay = base;
  for (let i = 0; i < 20; i += 1) {
    delay = nextPollDelay(delay, false, base, max, backoff);
  }
  assert.equal(delay, max);
});

test("sectionOf wraps into [0, sections) including negatives", () => {
  assert.equal(sectionOf(0), 0);
  assert.equal(sectionOf(8), 0);
  assert.equal(sectionOf(13), 5);
  assert.equal(sectionOf(-1), 7);
});

test("aggregateSections bins records, alerts, and finite peak THD", () => {
  const observations = [
    { source_index: 0, prediction: "anomaly", thd_ratio: 0.1 },
    { source_index: 0, prediction: "normal", thd_ratio: 0.04 },
    { source_index: 9, prediction: "anomaly", thd_ratio: Infinity },
    { source_index: 1, prediction: "normal", thd_ratio: 0.02 },
  ];
  const stats = aggregateSections(observations);
  assert.equal(stats.length, 8);
  assert.deepEqual(stats[0], { records: 2, alerts: 1, maxThd: 0.1 });
  // source_index 9 and 1 both land in section 1; non-finite THD is ignored.
  assert.equal(stats[1].records, 2);
  assert.equal(stats[1].alerts, 1);
  assert.equal(stats[1].maxThd, 0.02);
});

function resultFixture() {
  const decision = {
    sample_index: 0,
    source_index: 4,
    prediction: "normal",
    reconstruction_error: 0.01,
    thd_ratio: 0.02,
    triggers: [],
  };
  return {
    schema_version: 2,
    run: { id: "run-1", status: "complete" },
    config: { thd_limit: 0.05 },
    model: { parameters: 10 },
    training: {
      epochs: [{ epoch: 1, mse: 0.2, latency_ms: 1.0 }],
      calibration: { threshold: 0.1 },
    },
    evaluation: { observations: [{ ...decision }] },
    replay: {
      time_ms: [0, 1],
      records: [{
        ...decision,
        actual: [[0, 1], [0, 1], [0, 1]],
        reconstruction: [[0, 1], [0, 1], [0, 1]],
      }],
    },
  };
}

test("assertResultsShape validates bounded aligned waveform evidence", () => {
  const valid = resultFixture();
  assert.equal(assertResultsShape(valid), valid);
  const malformed = resultFixture();
  malformed.replay.records[0].actual[0] = [0];
  assert.throws(() => assertResultsShape(malformed), /invalid actual samples/);
  const oversizedHistory = resultFixture();
  oversizedHistory.training.epochs = Array.from(
    { length: 501 }, (_, index) => ({
      epoch: index + 1, mse: 0.1, latency_ms: 1,
    }),
  );
  assert.throws(() => assertResultsShape(oversizedHistory), /Training/);
});

test("dispatch cache identity includes both run ID and results digest", () => {
  const a = "a".repeat(64);
  const b = "b".repeat(64);
  assert.equal(dispatchCacheMatches("run-1", a, "run-1", a), true);
  assert.equal(dispatchCacheMatches("run-1", a, "run-1", b), false);
  assert.equal(dispatchCacheMatches("run-1", a, "run-2", a), false);
});

test("assertDispatchShape binds work orders to the exact displayed results", () => {
  const hash = "a".repeat(64);
  const artifact = {
    schema_version: 1,
    run_id: "run-1",
    results_sha256: hash,
    detected_assets: [],
    operational_dispatches: [{
      dispatch_id: "D-1",
      target_node: "N-1",
      priority: "HIGH",
      action_required: "Inspect",
    }],
  };
  assert.equal(assertDispatchShape(artifact, "run-1", hash), artifact);
  assert.throws(
    () => assertDispatchShape(artifact, "run-2", hash),
    /does not belong/,
  );
  assert.throws(
    () => assertDispatchShape(artifact, "run-1", "b".repeat(64)),
    /not bound/,
  );
});

// --- IEEE 1366 SAIFI/SAIDI classification tests ---

test("classifySaifi returns nominal for very low values", () => {
  assert.equal(classifySaifi(0.0005), "nominal");
});

test("classifySaifi returns warning for moderate values", () => {
  assert.equal(classifySaifi(0.003), "warning");
});

test("classifySaifi returns alarm for high values", () => {
  assert.equal(classifySaifi(0.01), "alarm");
});

test("classifySaifi returns invalid for non-finite input", () => {
  assert.equal(classifySaifi(NaN), "invalid");
  assert.equal(classifySaifi(-1), "invalid");
});

test("classifySaidi returns nominal for very low values", () => {
  assert.equal(classifySaidi(0.005), "nominal");
});

test("classifySaidi returns warning for moderate values", () => {
  assert.equal(classifySaidi(0.03), "warning");
});

test("classifySaidi returns alarm for high values", () => {
  assert.equal(classifySaidi(0.1), "alarm");
});

test("formatReliabilityRow formats SAIFI with correct unit", () => {
  const ri = { SAIFI: 0.006452, SAIDI: 2.5, CAIDI: 387.6, total_customer_interruptions: 8, customer_minutes_interrupted: 2500 };
  const row = formatReliabilityRow("SAIFI", ri);
  assert.equal(row.metric, "SAIFI");
  assert.equal(row.value, "0.006452");
  assert.equal(row.unit, "interruptions/customer");
});

test("formatReliabilityRow formats CAIDI with 2 decimal places", () => {
  const ri = { CAIDI: 387.6 };
  const row = formatReliabilityRow("CAIDI", ri);
  assert.equal(row.value, "387.60");
  assert.equal(row.unit, "minutes/interruption");
});

test("formatReliabilityRow formats integer count fields", () => {
  const ri = { total_customer_interruptions: 8, customer_minutes_interrupted: 2500 };
  const row = formatReliabilityRow("total_customer_interruptions", ri);
  assert.equal(row.value, "8");
  assert.equal(row.unit, "events");
});

// --- MPPT efficiency classification tests ---

test("classifyMpptEfficiency returns optimal above 97 percent", () => {
  assert.equal(classifyMpptEfficiency(0.98), "optimal");
});

test("classifyMpptEfficiency returns degraded between 90 and 97 percent", () => {
  assert.equal(classifyMpptEfficiency(0.93), "degraded");
});

test("classifyMpptEfficiency returns critical below 90 percent", () => {
  assert.equal(classifyMpptEfficiency(0.85), "critical");
});

test("classifyMismatch returns alarm above threshold", () => {
  assert.equal(classifyMismatch(0.6, 0.5), "alarm");
});

test("classifyMismatch returns warning at half threshold", () => {
  assert.equal(classifyMismatch(0.3, 0.5), "warning");
});

test("classifyMismatch returns nominal below half threshold", () => {
  assert.equal(classifyMismatch(0.1, 0.5), "nominal");
});

test("mpptSectionStatus returns is-ok for optimal sections", () => {
  assert.equal(mpptSectionStatus({ status: "optimal" }), "is-ok");
});

test("mpptSectionStatus returns is-warn for degraded sections", () => {
  assert.equal(mpptSectionStatus({ status: "degraded" }), "is-warn");
});

// --- DQ/PLL classification tests ---

test("classifyRipple returns alarm above threshold", () => {
  assert.equal(classifyRipple(0.06, 0.05), "alarm");
});

test("classifyRipple returns nominal below threshold", () => {
  assert.equal(classifyRipple(0.03, 0.05), "nominal");
});

test("classifyPllError returns alarm above threshold", () => {
  assert.equal(classifyPllError(6.0, 5.0), "alarm");
});

test("classifyPllError returns nominal below threshold", () => {
  assert.equal(classifyPllError(3.0, 5.0), "nominal");
});

test("pllLockStatus returns locked for locked sections", () => {
  const result = pllLockStatus({ pll_locked: true });
  assert.equal(result.text, "locked");
  assert.equal(result.className, "is-ok");
});

test("pllLockStatus returns UNLOCKED for unlocked sections", () => {
  const result = pllLockStatus({ pll_locked: false });
  assert.equal(result.text, "UNLOCKED");
  assert.equal(result.className, "is-alarm");
});

// --- Extraction function tests ---

test("extractReliabilityIndices returns null for missing data", () => {
  assert.equal(extractReliabilityIndices(null), null);
  assert.equal(extractReliabilityIndices({}), null);
  assert.equal(extractReliabilityIndices({ reliability_indices: {} }), null);
});

test("extractReliabilityIndices returns structured data with status", () => {
  const gr = {
    reliability_indices: {
      SAIFI: 0.006,
      SAIDI: 0.04,
      CAIDI: 6.67,
      total_customers_served: 1000,
      total_customer_interruptions: 6,
      customer_minutes_interrupted: 40,
    },
  };
  const result = extractReliabilityIndices(gr);
  assert.equal(result.SAIFI, 0.006);
  assert.equal(result.SAIDI, 0.04);
  assert.equal(result.saifiStatus, "alarm");
  assert.equal(result.saidiStatus, "warning");
  assert.equal(result.total_customers_served, 1000);
});

test("extractMpptMetrics returns null for missing data", () => {
  assert.equal(extractMpptMetrics(null), null);
  assert.equal(extractMpptMetrics({}), null);
});

test("extractMpptMetrics returns structured data with statuses", () => {
  const artifact = {
    mppt_metrics: {
      aggregate_mppt_efficiency: 0.9131,
      total_generation_kw: 1.2,
      total_mismatch_kw: 0.3,
      faulted_sections: ["SEC-02"],
      sections: [{ section: "SEC-00", mppt_efficiency: 0.98, status: "optimal" }],
    },
  };
  const result = extractMpptMetrics(artifact);
  assert.equal(result.aggregate_mppt_efficiency, 0.9131);
  assert.equal(result.efficiencyStatus, "degraded");
  assert.equal(result.mismatchStatus, "warning");
  assert.equal(result.faulted_sections.length, 1);
});

test("extractDqPllMetrics returns null for missing data", () => {
  assert.equal(extractDqPllMetrics(null), null);
  assert.equal(extractDqPllMetrics({}), null);
});

test("extractDqPllMetrics returns structured data with statuses", () => {
  const gr = {
    dq_pll_metrics: {
      aggregate_ripple_a: 0.08,
      aggregate_pll_error_deg: 3.5,
      unlocked_sections: ["SEC-03"],
      ripple_threshold_a: 0.05,
      pll_lock_threshold_deg: 5.0,
      sections: [{ section: "SEC-00", pll_locked: true }],
    },
  };
  const result = extractDqPllMetrics(gr);
  assert.equal(result.aggregate_ripple_a, 0.08);
  assert.equal(result.rippleStatus, "alarm");
  assert.equal(result.pllStatus, "nominal");
  assert.equal(result.unlocked_sections.length, 1);
});
