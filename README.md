# Edge-AI Smart Grid Anomaly Detector

A deterministic PyTorch framework for three-phase 60 Hz anomaly detection,
robustness experiments, and recorded grid-response simulation. The core model
combines PLL-normalized mirrored Conv1D reconstruction, a differentiable
Fortescue symmetrical-components penalty, and PLL-aware FFT total harmonic
distortion (THD); the browser console replays the exact exported evidence in
2D and 3D.

Live simulation console: <https://grid-sentinel-live.vercel.app/>

The console is a recorded synthetic replay, not live utility telemetry or a
production control interface.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python smart_grid_detector.py --export-all --multi-seed 10 --device cpu
```

Serve the generated console on the loopback interface:

```powershell
python smart_grid_detector.py --serve-dashboard-only
```

Open `http://127.0.0.1:8000/` and stop the server with `Ctrl+C`. The server
uses a strict public-file allowlist, blocks dotfiles/deployment metadata,
disables directory listing, and emits restrictive browser security headers.

## What the full experiment includes

- deterministic 10 kHz three-phase waveform generation with exactly 5%
  transients, voltage sags, and harmonic anomalies across 59.5–60.5 Hz;
- per-record phase-locked frequency tracking that normalizes drift before the
  autoencoder so reconstruction scoring is phase-reference stable;
- three-layer Conv1D/ConvTranspose1D autoencoder training on normal rows only;
- a custom Fortescue layer returning positive-, negative-, and zero-sequence
  RMS magnitudes, with negative/zero sequence regularization in the loss;
- mean + 3.5 population-standard-deviation reconstruction calibration;
- leakage-resistant, PLL-aware 2nd–50th harmonic THD and per-alert attribution;
- ten independent full-pipeline seeds with 95% confidence intervals;
- threshold sensitivity, tie-aware ROC/AUC, and adversarial boundary probes;
- drift monitoring and physics-consistency false-data-injection experiments;
- FP32/dynamic-int8 autoencoder size and forward-latency comparison plus a
  batch latency/throughput profile;
- synthetic feeder isolation, rerouting, and run-bound dispatch work orders;
- versioned JSON, a generated Markdown report, an H.264 explainer, and an
  accessible responsive dashboard with an optional Three.js topology view.

The latest primary seed recorded 100% accuracy, precision, recall, and F1 on
400 synthetic test waveforms (TP=20, TN=380, FP=0, FN=0). Across ten seeds,
mean accuracy was 99.95% and mean recall 99.50%. These are controlled synthetic
verification results, not field-performance claims.

## Generated artifacts

- `simulation_site/simulation_results.json` — canonical detector and experiment data
- `simulation_site/grid_dispatches.json` — dispatches bound to the exact run JSON hash
- `SIMULATION_RESULTS.md` — generated experiment report
- `simulation_site/smart_grid_explainer.mp4` — 18-second H.264 explainer
- `simulation_site/index.html` — recorded-replay console

The source SHA-256 is embedded in the canonical JSON. Regression tests fail if
the source, report data, dispatch binding, or video become stale.

## Verification

```powershell
python -m py_compile smart_grid_detector.py tests\test_smart_grid_detector.py
python -m unittest discover -s tests -p "test_*.py" -v
node --check simulation_site\app.js
node --check simulation_site\logic.mjs
node --check simulation_site\topology3d.js
node --test tests\test_frontend_logic.mjs
```

`scipy.signal` is used when installed; the detector retains a deterministic
Tukey-window fallback. The current dependency and vendored Three.js audits have
no known advisories.

## Project records

- `SECURITY_REVIEW.md` — current security scope, findings, fixes, and limits
- `CLAUDE_CODE_CHANGE_REVIEW.md` — trace and disposition of Claude Code's expansion
- `BLACKBOX_CHANGE_REVIEW.md` — earlier Blackbox change audit
- `PRODUCT.md` and `DESIGN.md` — product and interface decisions

Field deployment still requires utility telemetry, calibrated sensor and grid
physics models, hardware-in-loop tests, operational approval, and authenticated
control-system integrations. Self-healing actions here are simulations only.
