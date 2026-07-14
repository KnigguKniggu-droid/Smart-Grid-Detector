# Grid Sentinel - Project Handoff

## What this project is

A working Edge-AI Smart Grid Anomaly Detector. The core detector is a single Python file (`smart_grid_detector.py`) that synthesizes deterministic three-phase power waveforms, trains a PLL-normalized Conv1D autoencoder with Fortescue symmetrical-components loss, and raises alerts via two independent gates (reconstruction error + THD). The frontend is a fully static simulation dashboard at https://grid-sentinel-live.vercel.app/ that replays a recorded run from JSON artifacts.

## Canonical run

- Run ID: `20260713T042910Z-seed42`
- Python source SHA-256: `599a8daaad81e9dfbd0b6e5a4f4d5ed30d7bdedcffb81208d281e92d1f009041`
- 35 Python tests, 9 Node frontend tests, all passing
- Ruff lint clean

## Project structure

```
smart-grid-detector/
├── smart_grid_detector.py          # Single-file detector (all logic)
├── tests/
│   ├── test_smart_grid_detector.py # 35 Python tests (5 test classes)
│   └── test_frontend_logic.mjs     # 9 Node tests
├── simulation_site/                # Static dashboard (Vercel)
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   ├── logic.mjs                   # DOM-free clock, aggregation, schema validation
│   ├── dsp-engine.js               # Clarke Transform, FFT, phase portrait (NEW)
│   ├── autoencoder-sim.js          # Client-side autoencoder simulation (NEW)
│   ├── hardware-export.js          # C++ code generation for embedded (NEW)
│   ├── topology3d.js               # Three.js 3D grid topology
│   ├── topology-loader.js
│   ├── simulation_results.json     # Recorded run data
│   ├── simulation_results_alt.json # Comparison run (seed 137)
│   ├── grid_dispatches.json        # Self-healing dispatch data
│   ├── adversarial_resilience.json # 9 adversarial/sensor-failure scenarios
│   ├── smart_grid_explainer.mp4    # Generated from run artifacts
│   ├── SIMULATION_RESULTS.md
│   ├── vercel.json
│   └── favicon.svg
├── .github/workflows/ci.yml       # CI: Python tests, Node tests, dependency audit
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml                  # Ruff config (locked-source ignores)
├── opencode.json                   # OpenCode config
├── CLAUDE.md                       # Project rules + task-observer activation
├── AGENTS.md                       # Agent instructions
├── GOOGLE_AI_PROJECT_HANDOFF.md    # Authoritative continuation guide
├── SESSION_HANDOFF.md              # This file
└── SIMULATION_RESULTS.md
```

## Git and deployment

- Repo: `github.com/KnigguKniggu-droid/Smart-Grid-Detector` (public)
- Git user: `KnigguKniggu-droid` / `KnigguKniggu-droid@users.noreply.github.com`
- Branch protection on `main` requires: `Python tests`, `Node checks`, `Dependency audit`
- Vercel project: `<redacted>/grid-sentinel-live`
- Live URL: https://grid-sentinel-live.vercel.app/

## Core detector details

- 10 kHz three-phase synthesis, 5% anomaly rate, frequency drift +/-0.5 Hz
- PLL-normalized Conv1D autoencoder
- Fortescue symmetrical-components loss (alpha=0.01)
- PLL-aware FFT THD (5% limit)
- OR decision rule (either gate fires the alert)
- Deterministic seed: 42

## Test classes (35 Python tests)

1. `WaveformGenerationTests` - signal synthesis, anomaly injection, phase balance
2. `ModelAndFourierTests` - autoencoder, FFT, THD, Fortescue loss
3. `HardeningAndArtifactTests` - JSON export, determinism, hash verification
4. `RobustnessAndEdgeTests` - boundary probes, edge quantization, drift monitoring
5. `AdversarialAndSensorFailureTests` - 9 scenarios: stuck-at fault, phase swap, burst corruption, additive noise, sensor saturation, multi-phase bias, gain drift, replay masking, genuine-records guard

## What was built before this session (by Codex/Claude Code)

- Complete Python detector in `smart_grid_detector.py`
- All 26 original Python tests
- 9 Node frontend tests in `logic.mjs`
- Full dashboard: waveform replay, decision gate, training convergence, validation metrics, alert ledger, 3D topology, resilience suite, media section, provenance
- GitHub Actions CI pipeline
- Branch protection rules
- Vercel deployment
- `GOOGLE_AI_PROJECT_HANDOFF.md`
- `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`
- `opencode.json` with btw-skill

## What was done in this session

### 1. UI Revamp (styles.css)
- Glassmorphic dark theme with OKLCH color system
- Custom CSS design tokens (--bg, --surface, --cyan, --ink, etc.)
- System fonts only (CSP blocks external font loading)
- Focus states, skip-link accessibility, reduced-motion support

### 2. Adversarial Resilience Dashboard Section
- New test class `AdversarialAndSensorFailureTests` with 9 scenarios (total now 35 tests)
- Generated `adversarial_resilience.json` data file
- New HTML section in index.html between resilience suite and media section
- New CSS: adversarial cards, badges, summary stats
- New JS: `loadAdversarialResilience()` and `renderAdversarialResilience()` in app.js
- Updated `DASHBOARD_PUBLIC_PATHS` in smart_grid_detector.py
- Updated `vercel.json` with cache-control for the new JSON file
- All artifacts regenerated via `--export-all`

### 3. Mobile Responsive CSS
- Three-tier responsive layout: 1080px, 720px, 480px breakpoints
- 1080px: overview hero stacks, signal workspace single column, experiment grid single column
- 720px: topbar unstacks, metric-ledger/adversarial/resilience grids single column, pipeline flow 2-column, smaller headings
- 480px: pipeline flow single column, compact controls, smaller canvas heights, reduced typography, tighter padding
- Covers: hero, waveform plot, decision rail, pipeline, metric ledger, confusion table, adversarial cards, resilience cards, topology canvas, media section, provenance, footer

### 4. Self-Contained HTML Report Download
- "Download self-contained report" button in the provenance section
- `generateReport()` async function in app.js builds standalone HTML
- Includes: run metadata, validation metrics, confusion matrix, training info, alert ledger, adversarial resilience
- No external dependencies; works offline
- Button disabled until data loads, styled with cyan gradient

### 5. Multi-Run Comparison
- Run selector dropdown in topbar to switch between seed 42 and seed 137
- `simulation_results_alt.json` (seed 137) for comparison
- Comparison section with training/validation/calibration/alerts side-by-side
- `DASHBOARD_PUBLIC_PATHS` updated to include alt file
- `vercel.json` cache-control for alt file

### 6. Accessibility Audit
- `aria-live="polite"` on comparison and run selector
- `aria-describedby` on run selector
- Global keyboard nav (left/right arrows for records, spacebar for play/pause)
- Screen-reader-only comparison summary

### 7. Live Simulation Mode (Fault Injection)
- 8 fault types: amplitude sag/swell, phase offset, harmonic injection, Gaussian noise, DC offset, frequency drift
- Severity slider and phase selector
- Real-time reconstruction error and THD computation
- Waveform canvas redraws with injected faults
- Alarm/normal badge updates

### 8. Live DSP Engine (NEW)
- `dsp-engine.js`: Clarke Transform (abc to alpha-beta-zero), FFT-based THD, phase portrait renderer, three-phase signal generator
- `autoencoder-sim.js`: Client-side forward pass using Clarke Transform phase imbalance and harmonic analysis to estimate reconstruction error
- `hardware-export.js`: Generates platform-specific C++ code for STM32, Arduino, and ESP32
- Phase portrait canvas: real-time alpha-beta trajectory visualization showing circular path of balanced voltages and fault warping
- DSP metrics panel: phase imbalance, zero-sequence, harmonic ratio, dominant harmonic, decision gate badge
- Hardware deployment panel: platform selector, code generation, download, copy to clipboard
- All wired into the animation loop at 60fps

## Key things to know

- The source SHA-256 is locked by tests. If you modify `smart_grid_detector.py`, run `python smart_grid_detector.py --export-all` to regenerate artifacts, then run tests. The hash in the tests will need updating if the source changes.
- `pyproject.toml` ruff ignores `I001`, `UP017`, `W291` because source SHA-256 is locked.
- CSP headers in `vercel.json` block external font loading, so all fonts are system fonts.
- The 3D topology uses Three.js loaded from `topology-loader.js`.
- All JSON artifacts are fetched with `cache: no-store` to avoid stale data.
- `skill-observations/` folder is gitignored per CLAUDE.md rules.

## How to continue

1. Run all tests: `python -m pytest tests/test_smart_grid_detector.py -v` and `node tests/test_frontend_logic.mjs`
2. Lint: `python -m ruff check smart_grid_detector.py tests/test_smart_grid_detector.py`
3. Deploy: `npx vercel --prod --yes` from the `simulation_site/` directory
4. The dashboard auto-polls for new JSON artifacts every 5 seconds
5. After modifying `smart_grid_detector.py`, always run `--export-all` and update the source hash in tests
