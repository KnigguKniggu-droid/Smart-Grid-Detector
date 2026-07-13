# Product

## Register

product

## Platform

web

## Users

Grid operations engineers, power-quality analysts, edge-AI developers,
grid-security and resilience researchers, and technical reviewers use this
project to understand how three-phase anomalies are generated, detected,
measured, and reported, and how the detector holds up under adversarial and
distribution-shift conditions. They need to verify detector behavior quickly,
distinguish normal operation from transients, voltage sags, and harmonic
distortion, evaluate resilience against false-data-injection and drift, and
communicate repeatable experiment results.

## Product Purpose

The project is an executable reference framework for an Edge-AI smart-grid
detection-and-resilience pipeline, not only a single detector demo. It
combines deterministic synthetic data, PLL-normalized Conv1D reconstruction,
a differentiable Fortescue symmetrical-components loss, PLL-aware FFT-based
THD analysis, multi-seed robustness evaluation, adversarial
boundary probing, edge quantization benchmarking, drift monitoring,
false-data-injection resilience testing, a self-healing dispatch simulation,
an operational replay dashboard with a rendered 3D grid scene, and a
generated explainer video. Success means the same command can reproduce the
full experiment suite, emit trustworthy artifacts including honest negative
findings, and make the complete detection-and-response pipeline
understandable without inventing data in the interface.

## Brand Personality

Precise, operational, trustworthy. The experience should feel like a serious
power-quality instrument with enough visual energy to explain a live system, not
like a generic analytics template.

## Anti-references

Avoid generic SaaS metric-card grids, decorative glass effects, fake “live” data,
unreadable neon-on-black styling, mojibake, unexplained perfect scores, and static
dashboards that tell the user to make their own screen recording. Avoid visual
alarm states that depend on red/green color alone.

## Design Principles

1. Show the signal, then the decision: connect PLL tracking, waveform evidence,
   reconstruction error, THD, and the final alert.
2. Keep provenance visible: display the run timestamp, configuration, thresholds,
   and recorded-vs-live status.
3. Make abnormal conditions unmistakable without overwhelming normal operation.
4. Prefer measured data and honest limitations over decorative telemetry.
5. Keep the complete workflow runnable locally with minimal setup.

## Accessibility & Inclusion

Target WCAG 2.2 AA contrast and keyboard behavior. Support reduced motion, visible
focus, semantic landmarks and tables, screen-reader status updates, and redundant
shape/text cues for anomaly states so color-vision differences do not hide alerts.
