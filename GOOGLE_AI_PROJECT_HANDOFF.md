# Google AI Project Handoff — Edge-AI Smart Grid Anomaly Detector

Updated: 2026-07-12 (America/Chicago)  
Workspace: `C:\Users\ganes\Downloads\smart grid detector`  
Live console: <https://grid-sentinel-live.vercel.app/> (redeployment pending)

## Purpose of this document

This is the authoritative handoff for continuing the project with Google AI.
Read it before proposing or making changes. Inspect the current files directly
and preserve the verified controls, tests, generated-artifact contracts, and
recorded-vs-live labeling described below.

The project is currently operational. It is not an unfinished scaffold.

## Project objective

The system detects anomalies in synthetic three-phase grid-voltage waveforms
spanning 59.5–60.5 Hz using two independent signals:

1. reconstruction error from a PLL-normalized mirrored PyTorch Conv1D
   autoencoder;
2. PLL-aware total harmonic distortion calculated with `torch.fft.rfft`.

Training uses a custom differentiable Fortescue layer and the objective
`MSE + 0.01 × (negative-sequence RMS + zero-sequence RMS)`. This physics term
is a regularizer, not a third detection gate.

A waveform is flagged when reconstruction error exceeds its calibrated limit,
THD exceeds 5%, or the telemetry is invalid/non-finite. The framework also
records robustness experiments, physics-consistency checks, synthetic
self-healing dispatches, a browser replay console, and an H.264 explainer.

## Current verified state

| Item | Current value |
|---|---|
| Canonical run | `20260713T042910Z-seed42` |
| Python source SHA-256 | `599a8daaad81e9dfbd0b6e5a4f4d5ed30d7bdedcffb81208d281e92d1f009041` |
| Results JSON SHA-256 | `46db93aa09a58e4fb4505cdc6b165e842136c8b684424c13ea25e1f53cd83d7b` |
| Test observations | 400 |
| Primary-run accuracy | 100.00% |
| Primary-run precision | 100.00% |
| Primary-run recall | 100.00% |
| Primary-run F1 | 100.00% |
| Confusion matrix | TP=20, TN=380, FP=0, FN=0 |
| Independent seed runs | 10 |
| Ten-seed mean accuracy | 99.98% |
| Ten-seed mean recall | 99.50% |
| Generated dispatches | 20 |
| Python tests | 26/26 passing |
| Frontend tests | 9/9 passing |
| Physics-layer delta review | Complete, no remaining blocker |

These are controlled synthetic results. They are not evidence of field accuracy
or permission to control a real electrical grid.

## System architecture

### Python runtime

The complete detector and experiment engine lives in:

- `smart_grid_detector.py`

Major components appear in this order:

1. deterministic imports, constants, resource limits, and seed setup;
2. deterministic synthetic three-phase waveform generation at 10 kHz;
3. transient, voltage-sag, and harmonic anomaly injection into exactly 5% of
   the complete dataset;
4. a recurrence-based `PhaseLockedLoop` supporting 60 Hz ±0.5 Hz;
5. `SymmetricalComponentsLayer`, returning Fortescue positive-, negative-,
   and zero-sequence RMS magnitudes in that order;
6. `GridWaveformAutoencoder`, with three `Conv1d` encoder stages and three
   mirrored `ConvTranspose1d` decoder stages;
7. PLL-aware THD, harmonic attribution, and physics-informed training;
8. calibration, validation, metrics, and indexed alert logging;
9. multi-seed confidence intervals, threshold sweep, tie-aware ROC/AUC,
   boundary probes, latency profiling, drift monitoring, quantization, and
   false-data-injection experiments;
10. synthetic feeder isolation, rerouting, and dispatch generation;
11. strict JSON, Markdown report, H.264 video, and loopback HTTP serving.

### Browser console

The static site lives in `simulation_site/`:

- `index.html` — accessible control-room layout;
- `app.js` — results loading, replay UI, metrics, charts, alert evidence, and
  dispatch rendering;
- `logic.mjs` — DOM-free clock, aggregation, schema validation, and dispatch
  cache-identity logic shared with Node tests;
- `topology-loader.js` — lazy Three.js topology loading;
- `topology3d.js` — recorded synthetic feeder/substation visualization;
- `styles.css` — responsive visual system;
- `vercel.json` — production CSP and browser security headers;
- `vendor/three.module.min.js` — upstream Three.js revision 170;
- `simulation_results.json` — canonical experiment data;
- `grid_dispatches.json` — work orders bound to the exact run/results hash;
- `smart_grid_explainer.mp4` — generated 18-second H.264 explainer.

The public site is a read-only recorded replay. It is not a live telemetry or
grid-control endpoint.

## Detection and data rules that must remain invariant

- Waveforms use shape `[batch, 3, sequence_length]`.
- Default sequence length is 512 samples.
- Sample rate is 10,000 Hz and the fundamental is 60 Hz.
- Normal frequency drift spans 59.5–60.5 Hz and is PLL-normalized before
  encoding and reconstruction scoring.
- Synthetic anomaly prevalence is exactly 5% across the full dataset.
- Only normal training rows may fit the autoencoder or calibrate its threshold.
- The default reconstruction threshold is:
  `mean_training_error + 3.5 * population_standard_deviation`.
- THD above 5% is an independent anomaly trigger.
- The Fortescue training penalty uses negative- plus zero-sequence RMS with
  `alpha = 0.01`; it must not be presented as an inference-time alert gate.
- Invalid, non-finite, flatline, or missing-fundamental telemetry fails closed.
- Final prediction is the logical OR of reconstruction, THD, and invalid-input
  flags.
- Harmonic attribution must select the same ratio-based worst phase as THD.
- ROC calculations must group tied scores at one threshold.
- All allocation-heavy tensor paths must validate shape, sequence length, and
  total elements before FFT/model work.
- Multi-seed count must be validated before materializing the seed list.

## Artifact integrity contract

`simulation_results.json` is the canonical source for the report, dashboard,
video, and dispatch simulation.

Required integrity behavior:

- JSON serialization uses `allow_nan=False`.
- Non-finite evidence is exported as JSON `null`, while alert triggers retain
  the invalid-input reason.
- Writes use an exclusive same-directory temporary, flush/fsync, and atomic
  replacement.
- Results contain the exact SHA-256 of `smart_grid_detector.py`.
- `grid_dispatches.json` contains its schema version, run ID, source SHA, and
  exact SHA-256 of `simulation_results.json`.
- The browser independently hashes the fetched results and rejects mismatched
  dispatches.
- Dispatch cache identity includes both run ID and results digest.
- Tests must fail when source and generated artifacts become stale.

If `smart_grid_detector.py` changes, regenerate all artifacts before declaring
the work complete.

## Security posture

The current security review is in `SECURITY_REVIEW.md`.

Important retained controls:

- compound memory, dataset, model, batch, epoch, seed-sweep, FFT, video, and
  browser-response limits;
- strict loopback binding to `127.0.0.1` for the Python server;
- an exact public-asset allowlist rather than serving every file in
  `simulation_site/`;
- directory listing, dotfiles, `.env*`, `.vercel`, `vercel.json`, source files,
  undeclared files, and traversal paths are not served;
- production CSP, `nosniff`, referrer, opener/resource, frame, and permissions
  policies;
- fixed same-origin fetch targets and no `innerHTML`, `eval`, data-driven
  imports, or dynamic executable URLs;
- bounded nested JSON validation before browser state is assigned;
- no subprocess/shell execution, unsafe deserialization, database/query engine,
  archive extraction, authentication surface, upload path, or public mutation
  endpoint in first-party runtime code;
- the former local `VERCEL_OIDC_TOKEN` file was deleted and must not be
  recreated inside a served directory.

The prior pre-PLL sealed security snapshot digest was:

`codex-security-snapshot/v1:sha256:ff17b7c3ac13269f7c0a42f2af816389396044bf2e0facd37091fd0e437c9fc8`

It does not identify the current source snapshot. The current targeted delta
review and test evidence are recorded in `SECURITY_REVIEW.md`.

## Significant fixes already completed

Do not reintroduce these resolved problems:

- Blackbox's biased THD denominator, which underreported injected harmonics;
- stale source/report/video/dashboard provenance;
- loopback serving of `.env.local` and `.vercel/project.json`;
- compound model/data configurations capable of multi-GiB allocations;
- unbounded direct tensor-analysis APIs;
- strict JSON failure on correctly detected infinite THD;
- ROC/AUC dependence on row order when scores tie;
- attribution by absolute harmonic energy instead of THD ratio;
- fabricated flatline harmonic explanations;
- FDI reporting over all crafted rows instead of learned-detector evasions;
- fail-open missing FDI calibration keys;
- frontend module initialization race;
- shallow replay/training/ROC schema checks;
- dispatches not being bound to the exact results artifact;
- dispatch cache reuse when a run ID stayed the same but results changed;
- continuous paused 3D rendering, stale manual-step clocks, missing keyboard
  controls, mobile overflow, and no-WebGL summary gaps;
- missing security headers on the public Vercel deployment;
- unbounded `--multi-seed` list materialization before validation.

Detailed history is available in:

- `CLAUDE_CODE_CHANGE_REVIEW.md`
- `BLACKBOX_CHANGE_REVIEW.md`

## Reproduction commands

Create a clean environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the canonical ten-seed experiment and regenerate every artifact:

```powershell
python smart_grid_detector.py --export-all --multi-seed 10 --device cpu
```

Serve the current local console:

```powershell
python smart_grid_detector.py --serve-dashboard-only
```

Then open `http://127.0.0.1:8000/`.

Run verification:

```powershell
python -m py_compile smart_grid_detector.py tests\test_smart_grid_detector.py
python -m unittest discover -s tests -p "test_*.py" -v
node --check simulation_site\app.js
node --check simulation_site\logic.mjs
node --check simulation_site\topology-loader.js
node --check simulation_site\topology3d.js
node --test tests\test_frontend_logic.mjs
```

## Rules for continuing work

Before editing:

1. inspect the current source and generated provenance rather than assuming this
   document is newer;
2. preserve user changes because this folder has no Git history;
3. avoid exposing real credentials or placing `.env*` inside the served site;
4. keep all simulation and self-healing language explicitly synthetic;
5. add focused regression tests for every bug or security boundary changed.

Before handing work back:

1. run the Python and Node suites;
2. regenerate JSON, report, dispatch, and MP4 after any Python source change;
3. verify source SHA, results SHA, run ID, and dispatch binding;
4. fully decode the final MP4;
5. test valid local assets and confirm private/undeclared paths return 404;
6. redeploy `simulation_site/` if public files changed;
7. confirm the production hashes and security headers match local files;
8. update `README.md`, `SIMULATION_RESULTS.md`, and `SECURITY_REVIEW.md` when
   their recorded facts change.

## Sensible next advancements

The following are future projects, not missing pieces of the current demo:

- establish a Git repository and CI pipeline with Python, Node, dependency,
  artifact-provenance, and deployment checks;
- introduce a separately threat-modeled telemetry adapter with authentication,
  rate limits, backpressure, signed messages, privacy controls, and bounded
  queues;
- validate with utility or hardware-in-loop telemetry and clearly separated
  train/calibration/test time ranges;
- add model checkpoint signing and safe versioned loading if persistence is
  introduced;
- benchmark the full detector path on the intended edge hardware rather than
  treating dynamic-int8 autoencoder timing as a deployment benchmark;
- expand adversarial and sensor-failure tests while reporting residual evasions;
- add authenticated approval workflows before any simulated dispatch design is
  connected to an external operations system.

## Instruction to Google AI

Treat the current repository as a verified working baseline. Continue from it;
do not replace it with a new scaffold. Lead with a read-only review, state the
specific improvement being attempted, preserve the invariants above, implement
the smallest complete change, add tests, regenerate dependent artifacts, and
report exact commands and results. Never convert the recorded replay into a
claim of live grid monitoring or autonomous real-world control.
