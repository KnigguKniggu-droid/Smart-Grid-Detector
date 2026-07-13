# Security Review — Edge-AI Smart Grid Anomaly Detector

Reviewed locally and artifacts regenerated: 2026-07-12 (America/Chicago)  
Source SHA-256: `599a8daaad81e9dfbd0b6e5a4f4d5ed30d7bdedcffb81208d281e92d1f009041`  
Recorded run: `20260713T042910Z-seed42`  
Production: <https://grid-sentinel-live.vercel.app/> (this enhancement has not been redeployed)

## Outcome

No open reportable vulnerability remains in the reviewed local application.
The earlier audit did find a concrete local credential-
exposure path in Claude Code's handoff; it was reproduced, fixed, regression-
tested, and the credential file was removed. The public static deployment was
not changed during this physics-layer enhancement and may lag the local build.

## Reviewed boundaries

- the single-file Python detector, direct tensor APIs, training,
  FFT/THD and physics analysis, artifact writers, media generation, and local
  HTTP server;
- Python and Node regression suites;
- HTML, CSS, JavaScript modules, Three.js topology, generated JSON/Markdown,
  dispatch data, and MP4;
- Vercel configuration and live production responses;
- local Claude/Playwright and Vercel metadata for secret or path leakage;
- Python requirements and vendored Three.js 0.170.0 advisories.

The Python CLI is a trusted local-operator surface. The Vercel site is public
but static and read-only: it accepts no credentials or user data and performs
only fixed same-origin fetches. Future real telemetry, remote APIs, and grid
control integrations are outside this boundary and require their own auth,
rate limiting, message integrity, tenant isolation, and operational controls.

## Validated finding closed

### Local dashboard exposed a Vercel credential file (CWE-200)

Before the fix, starting the built-in server and requesting `/.env.local`
returned HTTP 200 and the complete local `VERCEL_OIDC_TOKEN`; requesting
`/.vercel/project.json` also returned 200. The server bound to loopback, which
reduced reachability, but any local HTTP client aware of the path could read
files placed under the public directory.

The handler now allows only the declared dashboard assets. Dotfiles,
deployment metadata, `vercel.json`, undeclared files, and encoded traversal
requests all return 404. Legitimate HTML, modules, data, report, license, and
video assets remain available with correct MIME types and headers. The expired
credential file was deleted after confirming Vercel CLI account authentication
did not depend on it.

Regression coverage creates a temporary public directory containing simulated
`.env.local` and `.vercel/project.json` files, proves they return 404, and proves
valid assets and cache-busting query strings still return 200.

## Additional hardening and correctness closures

- Combined model, data, batch, activation, and optimizer memory is capped;
  direct training, validation, THD, harmonic-attribution, and physics APIs
  share shape, sequence, and total-element limits.
- Dynamic PLL-aware THD and attribution bins are computed arithmetically in
  `O(batch × harmonic_orders)` memory, avoiding multi-GiB broadcast tensors at
  the maximum accepted batch size.
- The PLL validates its Nyquist band, preserves FP16/BF16/FP32/FP64 model
  compatibility, and the Fortescue penalty is measured without rotating away
  negative- or zero-sequence evidence.
- Flatline, missing-fundamental, and non-finite input fails closed while strict
  JSON exports schema-safe `null` evidence rather than aborting publication.
- Generated JSON is strict and atomically replaced; source provenance is
  embedded and tested. Dispatch data carries schema version, run ID, source
  SHA, and exact results-file SHA-256.
- Browser data is capped at 5 MiB while streaming, times out, receives strict
  nested shape/type/numeric bounds, and reaches only text/canvas sinks. There
  is no `innerHTML`, dynamic script URL, `eval`, or user-controlled fetch.
- The local server binds to `127.0.0.1`, disables listing, blocks traversal,
  uses an explicit public-file set, and sends CSP, nosniff, no-referrer, CORP,
  and no-store headers.
- Vercel sends equivalent CSP/nosniff/referrer/CORP/COOP/frame/permissions
  headers. Canonical JSON and dispatch data are `no-store`; sensitive and
  deployment paths return 404.
- Python code contains no subprocess/shell execution, dynamic evaluation,
  unsafe deserialization, SQL/query engine, archive extraction, inbound upload,
  authentication bypass, or non-loopback application server.

## Verification evidence

- `python -m unittest discover -s tests -p "test_*.py" -v`: 26/26 pass.
- `node --test tests/test_frontend_logic.mjs`: 9/9 pass.
- Python and all JavaScript modules compile/parse successfully.
- Generated H.264 video frame decode: pass (960×544 RGB frame; 203,391 bytes).
- Canonical local JSON SHA-256 is
  `46db93aa09a58e4fb4505cdc6b165e842136c8b684424c13ea25e1f53cd83d7b`;
  the dispatch artifact binds to that exact digest and run ID.
- Earlier production sensitive-path probes confirmed `.env.local`,
  `.vercel/project.json`, and `vercel.json` all return 404; redeployment and
  live hash verification remain a separate release step.
- `pip-audit` for `requirements.txt`: no known vulnerabilities.
- OSV query for vendored `three` 0.170.0: no known vulnerabilities; the file
  and MIT license match upstream.

The prior machine-readable completed scan bundle remains outside the source
tree. This document records the targeted security and resource re-review of
the new PLL/Fortescue delta and the complete regression results.

## Residual limitations

- Synthetic 100% primary-run metrics do not establish field safety or field
  accuracy. Ten-seed results and adversarial probes reduce variance but do not
  replace real telemetry or hardware-in-loop validation.
- The physics-consistency FDI layer records residual evasions and therefore is
  explicitly not a complete attack-prevention claim.
- Dynamic-int8 is an optional autoencoder-forward microbenchmark using the
  currently available PyTorch eager API, not a full detector benchmark.
- The directory has no Git history. Change attribution relies on hashes,
  timestamps, session evidence, and generated provenance.
- A future telemetry service or control integration would materially change
  the threat model and must not reuse this static-demo security assessment.
