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
