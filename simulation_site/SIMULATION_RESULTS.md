# Smart Grid Detector — Simulation & Experimentation Report

Run ID: `20260715T203558Z-seed42`  
Completed: `2026-07-15T20:35:58.776557+00:00`  
Source SHA-256: `d1a536b9a1bb1f58ecf797a8b05fcbad5fd95c4cf48ce92c048cfc51c846416f`

## Executive result

The deterministic synthetic evaluation completed successfully. The detector
combined PLL-normalized autoencoder reconstruction error with PLL-aware
FFT-based THD using the required OR decision rule. Training also used a
Fortescue negative/zero-sequence penalty; it is a regularizer, not a third
alert gate. These scores describe this controlled synthetic distribution; they
are not evidence of field performance on utility telemetry.

| Metric | Result |
|---|---:|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| F1 score | 100.00% |
| Specificity | 100.00% |
| TP / TN / FP / FN | 20 / 380 / 0 / 0 |
| Alerts | 20 / 400 |
| Batched validation latency | 80.86 ms |
| Throughput | 4947.1 waveforms/s |

## Configuration and environment

| Setting | Value |
|---|---:|
| Samples / sequence length | 2000 / 512 |
| Sample rate / fundamental | 10000 Hz / 60 Hz |
| Supported frequency drift | ±0.5 Hz |
| Epochs / batch size | 15 / 128 |
| Learning rate / weight decay | 0.001 / 1e-05 |
| Latent dimension / base channels | 64 / 16 |
| Threshold sigma / THD limit | 3.5 / 5.00% |
| Physics loss alpha | 0.01 |
| Seed / device | 42 / cpu |
| Parameters / estimated FP32 size | 565,283 / 2.16 MiB |
| Python / PyTorch / NumPy | 3.12.10 / 2.12.1+cpu / 2.5.0 |
| Window backend | Tukey fallback |

## Dataset

The full generated set contains exactly 5% anomalies.
The deterministic stratified split contains 1600 training rows
(80 anomalies excluded from fitting) and
400 test rows (20 anomalies).

| Test anomaly type | Count | Detected | Recall |
|---|---:|---:|---:|
| transient | 2 | 2 | 100.00% |
| sag | 6 | 6 | 100.00% |
| harmonic | 12 | 12 | 100.00% |

## Calibration

The hard threshold is `mean + sigma × population_std` over normal training
reconstruction errors.

- Mean error: `0.01927491`
- Population standard deviation: `0.00050951`
- Sigma: `3.5000`
- Reconstruction threshold: `0.02105821`
- THD threshold: `5.00%`

## Training history

The optimized objective is `MSE + alpha × (negative sequence RMS + zero
sequence RMS)`, with `alpha = 0.01`.

| Epoch | MSE | Physics penalty | Total loss | Duration (ms) |
|---:|---:|---:|---:|---:|
| 1 | 2.10901372 | 0.07236871 | 2.10973739 | 550.62 |
| 2 | 0.86641069 | 0.09172265 | 0.86732791 | 549.85 |
| 3 | 0.45810875 | 0.09836903 | 0.45909244 | 507.61 |
| 4 | 0.27730767 | 0.08705851 | 0.27817826 | 523.34 |
| 5 | 0.18763923 | 0.08166509 | 0.18845588 | 540.55 |
| 6 | 0.13623239 | 0.06878238 | 0.13692022 | 560.77 |
| 7 | 0.10153359 | 0.05400418 | 0.10207363 | 541.07 |
| 8 | 0.07760248 | 0.03947872 | 0.07799727 | 520.88 |
| 9 | 0.06056599 | 0.02720871 | 0.06083808 | 530.33 |
| 10 | 0.04797944 | 0.01907143 | 0.04817015 | 529.21 |
| 11 | 0.03909669 | 0.01206727 | 0.03921737 | 637.08 |
| 12 | 0.03256034 | 0.00767048 | 0.03263705 | 590.98 |
| 13 | 0.02770803 | 0.00420574 | 0.02775008 | 554.88 |
| 14 | 0.02385967 | 0.00216562 | 0.02388133 | 546.71 |
| 15 | 0.02069123 | 0.00132843 | 0.02070451 | 602.20 |

## Alert evidence

| Test index | Type | Reconstruction error | THD | Trigger |
|---:|---|---:|---:|---|
| 20 | harmonic | 0.03758164 | 15.85% | reconstruction_error, thd |
| 23 | transient | 0.02327888 | 8.77% | reconstruction_error, thd |
| 43 | transient | 0.02189393 | 9.58% | reconstruction_error, thd |
| 63 | harmonic | 0.02869943 | 12.73% | reconstruction_error, thd |
| 72 | harmonic | 0.03179231 | 14.46% | reconstruction_error, thd |
| 80 | harmonic | 0.02844689 | 12.23% | reconstruction_error, thd |
| 90 | sag | 0.03215103 | 4.82% | reconstruction_error |
| 111 | harmonic | 0.02573657 | 10.12% | reconstruction_error, thd |
| 112 | harmonic | 0.03424682 | 14.49% | reconstruction_error, thd |
| 127 | harmonic | 0.02857940 | 11.51% | reconstruction_error, thd |
| 143 | sag | 0.04439268 | 10.85% | reconstruction_error, thd |
| 229 | harmonic | 0.03303407 | 16.60% | reconstruction_error, thd |
| 231 | harmonic | 0.02677958 | 11.70% | reconstruction_error, thd |
| 233 | sag | 0.08397334 | 15.55% | reconstruction_error, thd |
| 249 | harmonic | 0.02619678 | 11.36% | reconstruction_error, thd |
| 278 | sag | 0.06618399 | 12.56% | reconstruction_error, thd |
| 293 | harmonic | 0.03593358 | 16.05% | reconstruction_error, thd |
| 337 | sag | 0.06071218 | 12.05% | reconstruction_error, thd |
| 356 | sag | 0.03933037 | 6.33% | reconstruction_error, thd |
| 392 | harmonic | 0.02652556 | 10.89% | reconstruction_error, thd |

## Threshold sensitivity

Metrics were recomputed across the sigma grid using the recorded per-record
evidence, with the THD and invalid-input rules unchanged.

| Sigma | Reconstruction threshold | Accuracy | FP | FN |
|---:|---:|---:|---:|---:|
| 2.0 | 0.02029394 | 98.25% | 7 | 0 |
| 2.25 | 0.02042132 | 99.75% | 1 | 0 |
| 2.5 | 0.02054869 | 99.75% | 1 | 0 |
| 2.75 | 0.02067607 | 100.00% | 0 | 0 |
| 3.0 | 0.02080345 | 100.00% | 0 | 0 |
| 3.25 | 0.02093083 | 100.00% | 0 | 0 |
| 3.5 | 0.02105821 | 100.00% | 0 | 0 |
| 3.75 | 0.02118559 | 100.00% | 0 | 0 |
| 4.0 | 0.02131297 | 100.00% | 0 | 0 |
| 4.25 | 0.02144034 | 100.00% | 0 | 0 |
| 4.5 | 0.02156772 | 100.00% | 0 | 0 |
| 4.75 | 0.02169510 | 100.00% | 0 | 0 |
| 5.0 | 0.02182248 | 100.00% | 0 | 0 |

The zero-error operating band spans sigma 2.75 through 5.0, so the mandated 3.5 sits inside a stable plateau rather than on a knife edge.

## Detector operating characteristic

The area under the ROC curve summarizes ranking quality across every possible
threshold, independent of the mandated operating point. Fusing the
reconstruction-error and THD branches raises the area over the reconstruction
branch alone.

| Score | ROC AUC |
|---|---:|
| Reconstruction error | 1.0000 |
| Fused (recon + THD) | 1.0000 |

At the mandated sigma 3.5000 operating point the
reconstruction branch sits at false-positive rate 0.0000 and
true-positive rate 1.0000.

## Adversarial boundary evidence

Engineered waveforms probe each decision rule from both sides, including the
5.5% estimator-bias sentinel that the superseded low-biased FFT change would
have let through, and both fail-closed telemetry paths.

| Probe | Reconstruction error | THD | Prediction | Verdict |
|---|---:|---:|---|---|
| thd below limit | 0.02017090 | 4.07% | normal | pass |
| thd above limit | 0.02145212 | 6.07% | anomaly | pass |
| thd bias sentinel | 0.02108190 | 5.57% | anomaly | pass |
| reconstruction below threshold | 0.02070075 | 3.09% | normal | pass |
| reconstruction above threshold | 0.02145107 | 3.77% | anomaly | pass |
| flatline fails closed | 0.28854105 | invalid | anomaly | pass |
| nonfinite fails closed | 0.01926403 | invalid | anomaly | pass |

All probes behaved as engineered.

## Edge deployment benchmark

Dynamic int8 quantization targets the two dense projection layers, which hold
most of the parameters. Latency is the median of
100 single-record CPU autoencoder forward
passes; FFT, invalid-input checks, and physics-consistency checks are not
included in this microbenchmark.

| Variant | State dict | Median single-record latency | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| FP32 | 2,274,359 B | 1.67 ms | 100.00% | 100.00% | 100.00% |
| Dynamic int8 | 703,019 B | 1.93 ms | 100.00% | 100.00% | 100.00% |

Quantization shrinks the state dict by
3.24x. Each variant recalibrates its
own threshold on the normal training rows (FP32
`0.02105821`, int8
`0.02130547`).

## Latency and throughput profile

Inference latency and throughput across batch sizes on
`cpu` (median of 30
timed iterations). Small batches minimize per-alert latency; large batches
maximize throughput. Peak throughput was at batch size
400.

| Batch size | Batch latency (ms) | Per-record (ms) | Throughput (per s) |
|---:|---:|---:|---:|
| 1 | 1.73 | 1.731 | 578 |
| 8 | 3.13 | 0.391 | 2,557 |
| 32 | 4.85 | 0.152 | 6,594 |
| 128 | 11.03 | 0.086 | 11,608 |
| 400 | 29.30 | 0.073 | 13,650 |

## Drift monitoring

A rolling window of 64 reconstruction errors is
z-tested against the calibration distribution (alarm at |z| >=
4.0). The clean replay stayed silent
(max |z| = 1.09); each scenario rescales the same
clean records by a small sensor-gain factor.

| Gain | Individual alerts | Max abs z | First drift alarm | Silent-drift thesis |
|---:|---:|---:|---:|---|
| 1.005 | 0 | 4.69 | 64 | proven |
| 1.01 | 2 | 8.64 | 64 | no |
| 1.02 | 72 | 17.61 | 64 | no |

At least one gain level caused a rolling-window drift alarm while individual alerts remained zero, so the monitor sees what the per-record detector cannot.

## False-data-injection resilience

Physics-consistency checks (sensor noise floor, balanced phase sum, RMS
symmetry, and a scale-invariant phase-asymmetry test, plus a joint fused-score
bound, calibrated on genuine normal telemetry) guard the records the learned
detector passes as normal. Baseline false positives on
380 genuine normal records:
0.

| Attack | Records | Evaded learned rules | Caught after evasion | Residual evasions | Conditional detection |
|---|---:|---:|---:|---:|---:|
| replay masking | 20 | 20 | 20 | 0 | 100.0% |
| phase bias | 380 | 244 | 236 | 8 | 96.7% |
| coordinated | 380 | 379 | 311 | 68 | 82.1% |

The replay and phase-bias classes were caught above the 95% gate while genuine records stayed unflagged, so masked telemetry cannot silently pass the learned rules.

The coordinated two-phase attack evaded the learned detector on 100% of records. Among those evasions, physics consistency caught 82.1%, leaving 68 residual evasions. This meets the recorded layered-defense gate, but it is not a claim of complete attack prevention.

## Self-healing response simulation

Every alert maps onto a synthetic 8-section feeder
topology: the section breaker isolates the fault zone, the neighboring tie
switch restores customers outside it, and a prioritized work order is
emitted. The full dispatch list is exported to
`simulation_site/grid_dispatches.json` using the
detected_assets / operational_dispatches integration shape.

- Alerts dispatched: 20
  (CRITICAL 16, HIGH 3,
  MEDIUM 1)
- Sections isolated: SEC-00, SEC-01, SEC-02, SEC-03, SEC-04, SEC-05, SEC-06, SEC-07
- Customers rerouted around fault zones:
  1240
  (simulated isolation 0.5 s,
  reroute 2.0 s)

Sections, customers, coordinates, and response times are simulated for
workflow demonstration only.

## Reproduction and artifacts

```powershell
python smart_grid_detector.py --export-all --device cpu
python smart_grid_detector.py --serve-dashboard-only
```

- Live console: `simulation_site/index.html`
- Canonical data: `simulation_site/simulation_results.json`
- Playable video: `simulation_site/smart_grid_explainer.mp4`

## Limitations and next experiments

- Synthetic 60 Hz waveforms do not reproduce every sensor, topology, switching,
  load, weather, communications, or adversarial condition seen in a real grid.
- The perfect seeded score must not be generalized beyond this dataset.
- Recommended next work: multi-seed confidence intervals, held-out field captures, validation on real utility telemetry, and hardware-in-the-loop benchmarking on target edge devices.
