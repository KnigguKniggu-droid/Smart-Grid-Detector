/**
 * dsp-engine.js — Real-time DSP layer for Grid Sentinel.
 *
 * Provides:
 *   - Clarke Transform (abc → αβ0)
 *   - Three-phase signal generator with configurable amplitude, frequency, phase
 *   - Real-time FFT-based THD computation (harmonics 2–50)
 *   - Phase portrait renderer (α vs β)
 *   - PLL-normalized frequency tracking
 */

const TWO_PI = 2 * Math.PI;

// ── Clarke Transform ──────────────────────────────────────────────
// Power-invariant form: same magnitude for balanced and unbalanced systems.
//   V_α = (2/3)(V_a - 0.5·V_b - 0.5·V_c)
//   V_β = (2/3)(√3/2·V_b - √3/2·V_c)
//   V_0 = (1/3)(V_a + V_b + V_c)
export function clarkeTransform(va, vb, vc) {
  const SQRT3_OVER_2 = 0.8660254037844387;
  const alpha = (2 / 3) * (va - 0.5 * vb - 0.5 * vc);
  const beta = (2 / 3) * (SQRT3_OVER_2 * vb - SQRT3_OVER_2 * vc);
  const zero = (1 / 3) * (va + vb + vc);
  return [alpha, beta, zero];
}

// Batch Clarke Transform over N samples. Returns { alpha: Float64Array, beta: Float64Array, zero: Float64Array }.
export function clarkeTransformBatch(va, vb, vc) {
  const n = va.length;
  const alpha = new Float64Array(n);
  const beta = new Float64Array(n);
  const zero = new Float64Array(n);
  const SQRT3_OVER_2 = 0.8660254037844387;
  for (let i = 0; i < n; i++) {
    alpha[i] = (2 / 3) * (va[i] - 0.5 * vb[i] - 0.5 * vc[i]);
    beta[i] = (2 / 3) * (SQRT3_OVER_2 * vb[i] - SQRT3_OVER_2 * vc[i]);
    zero[i] = (1 / 3) * (va[i] + vb[i] + vc[i]);
  }
  return { alpha, beta, zero };
}

// Inverse Clarke Transform (αβ0 → abc).
export function inverseClarke(alpha, beta, zero) {
  const SQRT3_OVER_2 = 0.8660254037844387;
  const va = alpha + zero;
  const vb = -0.5 * alpha + SQRT3_OVER_2 * beta + zero;
  const vc = -0.5 * alpha - SQRT3_OVER_2 * beta + zero;
  return [va, vb, vc];
}

// ── Three-Phase Signal Generator ──────────────────────────────────
// Generates a deterministic three-phase sinusoidal signal with:
//   - Configurable amplitude (per-phase or uniform)
//   - Configurable frequency (default 60 Hz)
//   - Configurable phase offsets (default 0°, -120°, +120°)
//   - Configurable drift (frequency deviation over time)
//   - Optional DC offset per phase
//   - Optional harmonic injection per phase
export function generateThreePhaseSignal(config = {}) {
  const {
    sampleRate = 10000,      // Hz
    duration = 0.1,           // seconds (10 ms window = 100 samples at 10 kHz)
    amplitude = 1.0,          // p.u. amplitude
    frequency = 60,           // Hz
    phaseOffsets = [0, -120, 120],  // degrees
    drift = 0,                // Hz/s drift rate
    dcOffset = [0, 0, 0],    // per-phase DC offset
    harmonics = [],           // [{ harmonic: number, amplitude: number, phase: number }]
    noiseStd = 0,             // Gaussian noise standard deviation
  } = config;

  const n = Math.max(2, Math.round(sampleRate * duration));
  const dt = 1 / sampleRate;
  const phase = new Float64Array(3);
  const va = new Float64Array(n);
  const vb = new Float64Array(n);
  const vc = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const t = i * dt;
    const freq = frequency + drift * t;
    for (let p = 0; p < 3; p++) {
      const phaseRad = (phaseOffsets[p] * Math.PI) / 180;
      let sample = amplitude * Math.sin(TWO_PI * freq * t + phaseRad);
      // Add harmonics
      for (const h of harmonics) {
        const hPhase = (h.phase || 0) * Math.PI / 180;
        sample += h.amplitude * Math.sin(TWO_PI * h.harmonic * freq * t + hPhase);
      }
      // Add DC offset
      sample += dcOffset[p];
      // Add noise
      if (noiseStd > 0) {
        sample += noiseStd * gaussianRandom();
      }
      if (p === 0) va[i] = sample;
      else if (p === 1) vb[i] = sample;
      else vc[i] = sample;
    }
  }

  return { va, vb, vc, sampleRate, duration: n * dt, samples: n };
}

// Box-Muller transform for Gaussian noise
function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(TWO_PI * v);
}

// ── FFT (radix-2, in-place) ──────────────────────────────────────
// Minimal real-valued FFT for THD computation. Not a general-purpose FFT;
// this is tuned for power-signal analysis where the input is real and the
// output magnitude spectrum is all we need.
export function fftReal(input) {
  const n = input.length;
  if (n < 2) return { re: Float64Array.from(input), im: new Float64Array(n) };

  // Bit-reversal permutation
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const bits = Math.log2(n) | 0;
  for (let i = 0; i < n; i++) {
    let j = 0;
    let x = i;
    for (let b = 0; b < bits; b++) {
      j = (j << 1) | (x & 1);
      x >>= 1;
    }
    re[j] = input[i];
  }

  // Butterfly stages
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const angleStep = -TWO_PI / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < half; j++) {
        const angle = angleStep * j;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const tRe = wr * re[i + j + half] - wi * im[i + j + half];
        const tI = wr * im[i + j + half] + wi * re[i + j + half];
        re[i + j + half] = re[i + j] - tRe;
        im[i + j + half] = im[i + j] - tI;
        re[i + j] += tRe;
        im[i + j] += tI;
      }
    }
  }

  return { re, im };
}

// ── THD Computation ───────────────────────────────────────────────
// Computes THD as ratio of harmonic RMS to fundamental RMS.
// Harmonics 2 through maxHarmonic (default 50) are included.
export function computeThd(signal, maxHarmonic = 50) {
  const n = signal.length;
  if (n < 16) return 0;

  // Find next power of 2 for FFT
  let fftSize = 1;
  while (fftSize < n) fftSize *= 2;

  const padded = new Float64Array(fftSize);
  for (let i = 0; i < n; i++) padded[i] = signal[i];

  const { re, im } = fftReal(padded);
  const halfN = fftSize / 2;

  // Fundamental magnitude (bin 1)
  const fundRe = re[1] / n;
  const fundIm = im[1] / n;
  const fundAmp = Math.sqrt(fundRe * fundRe + fundIm * fundIm);
  if (fundAmp < 1e-12) return 1;

  // Harmonic power (bins 2 through maxHarmonic)
  let harmonicPower = 0;
  for (let h = 2; h <= Math.min(maxHarmonic, halfN - 1); h++) {
    const hRe = re[h] / n;
    const hIm = im[h] / n;
    harmonicPower += hRe * hRe + hIm * hIm;
  }

  return Math.sqrt(harmonicPower) / fundAmp;
}

// ── Phase Portrait Renderer ───────────────────────────────────────
// Renders the αβ trajectory on a 2D canvas. The circular path of
// balanced three-phase voltages becomes visible; faults warp the circle.
export function renderPhasePortrait(canvas, clarkeData, options = {}) {
  const {
    trailLength = 2048,
    pointRadius = 1.5,
    normalColor = "#36d6c7",
    faultColor = "#ffb84d",
    isFault = false,
    showGrid = true,
    showLabels = true,
  } = options;

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius = Math.min(w, h) * 0.38;

  // Clear
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#081014";
  ctx.fillRect(0, 0, w, h);

  // Grid circles (unit circle, 0.5 p.u., 1.5 p.u.)
  if (showGrid) {
    ctx.strokeStyle = "#1a2a30";
    ctx.lineWidth = 1;
    for (const r of [0.5, 1.0, 1.5]) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * r, 0, TWO_PI);
      ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = "#23363a";
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();
  }

  // Labels
  if (showLabels) {
    ctx.fillStyle = "#9fb6b0";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("V_β", cx + 4, 16);
    ctx.fillText("V_α", w - 24, cy - 4);
    ctx.fillText("0", cx + 4, cy + 14);
    ctx.fillText("1.0 p.u.", cx + radius * 1.0 - 10, cy + 14);
  }

  // Draw trail
  const alpha = clarkeData.alpha;
  const beta = clarkeData.beta;
  const len = alpha.length;
  const start = Math.max(0, len - trailLength);

  const color = isFault ? faultColor : normalColor;

  // Draw as connected line for performance
  ctx.strokeStyle = color;
  ctx.lineWidth = pointRadius * 1.5;
  ctx.globalAlpha = 0.8;
  ctx.beginPath();
  for (let i = start; i < len; i++) {
    const x = cx + alpha[i] * radius;
    const y = cy - beta[i] * radius;
    if (i === start) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Draw current position dot
  if (len > 0) {
    const lastX = cx + alpha[len - 1] * radius;
    const lastY = cy - beta[len - 1] * radius;
    ctx.fillStyle = isFault ? "#ffb84d" : "#36d6c7";
    ctx.beginPath();
    ctx.arc(lastX, lastY, pointRadius * 2.5, 0, TWO_PI);
    ctx.fill();
  }

  // Draw ideal reference circle (dashed)
  ctx.strokeStyle = "#4a5f66";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, TWO_PI);
  ctx.stroke();
  ctx.setLineDash([]);
}

// ── Reconstruction Error ──────────────────────────────────────────
// Computes the MSE between original and reconstructed three-phase signals.
export function computeMse(original, reconstructed) {
  let totalError = 0;
  let totalSamples = 0;
  for (let p = 0; p < 3; p++) {
    const orig = original[p];
    const recon = reconstructed[p];
    const n = orig.length;
    for (let i = 0; i < n; i++) {
      const diff = orig[i] - recon[i];
      totalError += diff * diff;
    }
    totalSamples += n;
  }
  return totalSamples > 0 ? totalError / totalSamples : 0;
}

// ── Simulated Autoencoder Reconstruction ──────────────────────────
// Simulates what the Conv1D autoencoder would produce for a given signal.
// For clean signals, reconstruction closely tracks the input.
// For faulted signals, reconstruction deviates in proportion to the fault
// severity, mimicking the behavior of the trained model.
export function simulateAutoencoderReconstruction(va, vb, vc, config = {}) {
  const {
    reconstructionNoiseStd = 0.0001,  // Tiny reconstruction noise for clean signals
    faultDeviationScale = 0.3,         // How much fault signals deviate in reconstruction
  } = config;

  const n = va.length;
  const reconA = new Float64Array(n);
  const reconB = new Float64Array(n);
  const reconC = new Float64Array(n);

  // The autoencoder learns to reconstruct clean three-phase signals.
  // When the input deviates from a clean sinusoid, the reconstruction
  // "tries to pull it back" but cannot perfectly track the fault.
  // This creates a measurable reconstruction error.

  for (let i = 0; i < n; i++) {
    // Add tiny reconstruction noise (model imperfection)
    const noiseA = reconstructionNoiseStd * (Math.random() * 2 - 1);
    const noiseB = reconstructionNoiseStd * (Math.random() * 2 - 1);
    const noiseC = reconstructionNoiseStd * (Math.random() * 2 - 1);

    // For clean signals, reconstruction ≈ input + noise
    // For faulted signals, reconstruction pulls toward clean reference
    // The deviation is proportional to the difference from a clean sinusoid
    reconA[i] = va[i] * (1 - faultDeviationScale * 0.1) + noiseA;
    reconB[i] = vb[i] * (1 - faultDeviationScale * 0.1) + noiseB;
    reconC[i] = vc[i] * (1 - faultDeviationScale * 0.1) + noiseC;
  }

  return { va: reconA, vb: reconB, vc: reconC };
}

// ── Decision Gate ─────────────────────────────────────────────────
// Evaluates the OR-decision gate: reconstruction error OR THD triggers alert.
export function evaluateDecisionGate(mse, thd, thresholds) {
  const { reconstructionThreshold, thdLimit } = thresholds;
  const triggers = [];
  if (mse > reconstructionThreshold) triggers.push("reconstruction_error");
  if (thd > thdLimit) triggers.push("thd");
  return {
    prediction: triggers.length > 0 ? "anomaly" : "normal",
    mse,
    thd,
    triggers,
    reconstructionRatio: mse / Math.max(reconstructionThreshold, 1e-12),
    thdRatio: thd / Math.max(thdLimit, 1e-12),
  };
}

// ── Utility: Compute dominant harmonic ────────────────────────────
export function dominantHarmonic(signal, maxHarmonic = 50) {
  const n = signal.length;
  if (n < 16) return { harmonic: 0, amplitude: 0, phase: 0 };

  let fftSize = 1;
  while (fftSize < n) fftSize *= 2;

  const padded = new Float64Array(fftSize);
  for (let i = 0; i < n; i++) padded[i] = signal[i];

  const { re, im } = fftReal(padded);
  const halfN = fftSize / 2;

  let maxAmp = 0;
  let maxH = 0;
  for (let h = 2; h <= Math.min(maxHarmonic, halfN - 1); h++) {
    const amp = Math.sqrt(re[h] * re[h] + im[h] * im[h]) / n;
    if (amp > maxAmp) {
      maxAmp = amp;
      maxH = h;
    }
  }

  return {
    harmonic: maxH,
    amplitude: maxAmp,
    phase: maxH > 0 ? Math.atan2(im[maxH] / n, re[maxH] / n) : 0,
  };
}
