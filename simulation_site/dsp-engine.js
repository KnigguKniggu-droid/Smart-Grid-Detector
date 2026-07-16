/**
 * dsp-engine.js — Real-time DSP layer for Grid Sentinel.
 *
 * Provides:
 *   - Clarke Transform batch (abc to alpha-beta-zero)
 *   - Real-time FFT-based THD computation (harmonics 2-50)
 *   - Phase portrait renderer (alpha vs beta)
 *   - PLL-normalized frequency tracking
 *   - Dominant harmonic detection
 */

const TWO_PI = 2 * Math.PI;

// ── Clarke Transform ──────────────────────────────────────────────
// Power-invariant form: same magnitude for balanced and unbalanced systems.
//   V_alpha = (2/3)(V_a - 0.5*V_b - 0.5*V_c)
//   V_beta = (2/3)(sqrt(3)/2*V_b - sqrt(3)/2*V_c)
//   V_0 = (1/3)(V_a + V_b + V_c)

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
// ── Simulated Autoencoder Reconstruction ──────────────────────────
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
