# Blackbox Change Review

## Traceability

This directory is not a Git repository and contained no older source snapshot, so
exact author-level attribution was not recoverable. The change trail was
reconstructed from the generated transcript, file timestamps, code semantics, and
analytical tests before the transcript was removed from the release surface.

## What was added

Blackbox attempted four useful advancements:

- configurable reconstruction-threshold sigma with a default of 3.5;
- JSON metric export for a browser dashboard;
- a polling HTML summary;
- a manually written experiment report.

The threshold option and export intent were retained. Their implementation was
hardened and expanded.

## What was replaced

### Biased THD denominator

The changed FFT code summed three correlated zero-padded bins for fundamental
power while using one bin for each harmonic. This biased THD low:

| Analytical waveform | Changed result | Corrected result |
|---|---:|---:|
| Pure 60 Hz | 0.128% | 0.220% |
| 5% third harmonic | 2.947% | 5.071% |
| 10% third harmonic | 5.856% | 10.076% |

The autoencoder happened to rescue the seeded test set, but a waveform relying on
the THD branch alone could evade the mandated 5% rule. The implementation now uses
the same single-bin estimator for fundamental and harmonic amplitudes and treats
absent fundamental power as invalid/infinite THD.

### Partial, non-atomic JSON

The original JSON contained aggregates only and was written directly to the final
path. The replacement uses a schema-versioned artifact, strict finite JSON,
same-directory temporary writes plus `os.replace`, all 400 scalar decisions,
training history, calibration statistics, anomaly types, selected actual and
reconstructed waveforms, and source/runtime provenance.

### Static dashboard labeled “live”

The original page polled an unchanged aggregate file and could not replay a
waveform or explain an alert. It has been replaced by an animated recorded-run
console with real three-phase and reconstruction traces, decision gates, training
loss, calibration, metrics, alert navigation, provenance, and accessible controls.

### Missing video and drifting report

No video existed; the report instructed the user to screen-record manually. The
framework now generates an H.264 MP4 directly from measured run data and generates
the Markdown report from the same canonical JSON, preventing copied metrics and
latency from drifting apart.

## Additional hardening

- centralized finite-value and resource-budget validation before allocation;
- deterministic seed propagation into data generation;
- loopback-only serving rooted at `simulation_site`;
- disabled directory listing and restrictive CSP, MIME, referrer, and resource
  headers;
- strict DOM rendering with `textContent` and version/range checks;
- explicit synthetic-data limitations in the report, dashboard, and video.

