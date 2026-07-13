# Claude Code Change Review

Reviewed: 2026-07-12 (America/Chicago)

## What Claude Code was building

Claude reconstructed the prior project state from local session artifacts and
expanded the detector from a single-run demonstration into a resilience and
operations framework. Its additions included ten-seed confidence intervals,
threshold and ROC analysis, decision-boundary probes, quantization and latency
benchmarks, drift monitoring, false-data-injection experiments, harmonic
explanations, a synthetic self-healing dispatch flow, and a Three.js feeder
topology. It also linked and deployed the static site to Vercel.

Those directions were retained. They materially improve the project and are
now represented in the regenerated report, JSON artifacts, video, and live
console.

## Problems found in the handoff

The review read all current Python, test, frontend, configuration, generated,
and deployment files. It reproduced these issues before fixing them:

- the Python source and local `index.html`/`app.js` had changed after the last
  export and Vercel deployment, leaving the report, JSON, video, and live UI
  stale;
- the local Python server served `simulation_site/.env.local` and
  `.vercel/project.json`, exposing a Vercel OIDC credential and metadata to any
  local HTTP client that knew the path;
- ROC/AUC advanced one record at a time through tied scores, making AUC depend
  on label order instead of ranking quality;
- harmonic attribution chose a phase by absolute harmonic energy rather than
  the detector's THD ratio and fabricated a harmonic for flatline input;
- new FFT and physics helper APIs lacked the established shape/size boundary;
- FDI conclusions used detection over all crafted rows instead of detection
  among rows that actually evaded the learned detector;
- the browser had an ES-module/classic-script initialization race, shallow
  nested schema checks, and no cryptographic binding between detector results
  and dispatch work orders;
- the public Vercel deployment lacked the local server's CSP and other security
  headers;
- the no-WebGL accessibility fallback, narrow mobile layout, paused rendering,
  manual-step clock, keyboard controls, and WebGL restore lifecycle needed
  completion.

## Disposition

All items above were fixed and regression-tested. The server now serves only a
declared public asset set. Results and dispatches are schema-validated and bound
by run ID plus SHA-256. ROC is tie-aware, attribution follows THD ratio,
exceptional telemetry fails closed, and FDI reporting includes caught-after-
evasion and residual-evasion counts. The browser is module-native, bounded,
keyboard-operable, responsive at 320 px, and on-demand while paused. Vercel
headers mirror the local security policy.

The generated artifacts were rebuilt from the final source with ten independent
seeds. The current site was then prepared for a fresh production deployment.

## Remaining limits

There is no Git history in this folder, so attribution relies on file times,
Claude/Playwright records, deployment hashes, and artifact provenance rather
than commits. Dynamic-int8 uses PyTorch's currently available eager API and is
an optional autoencoder-forward microbenchmark; it is not a full detector or
hardware benchmark. The FDI and self-healing layers are synthetic experiments,
not security or operational guarantees for a real grid.
