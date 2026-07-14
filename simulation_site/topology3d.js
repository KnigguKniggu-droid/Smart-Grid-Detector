import * as THREE from "./vendor/three.module.min.js";
import { sectionOf, aggregateSections } from "./logic.mjs";

// Daylight-realistic 3D grid scene. Three.js is vendored same-origin
// (vendor/) because the dashboard is served under a default-src 'self' CSP.
// Every texture is generated procedurally on canvases so the site stays
// fully self-contained: no model files, no image downloads.
//
// Realism pass (recorded-run overlay unchanged):
// - Transformers: ribbed tanks with corner stiffeners, bolted lids, radiator
//   banks with header pipes, shed-profile porcelain bushings, conservator
//   drum, valves, gauges and nameplates.
// - Poles: tapered trunks, wood crossarms with steel V-braces and through
//   bolts, pin insulators with real shed profiles, pole-top center phase,
//   guy wires on the dead-end poles.
// - Substation: lattice-braced gantries, suspension insulator strings on the
//   feeder takeoff structures, chain-link fence with barbed wire and a gate.
// - Conductors: catenary sag rendered as shaded tubes instead of flat lines.
// - Environment: layered grass / gravel / dirt-road textures with bump maps,
//   tree line, warm key light plus cool fill tuned for depth.
// Detection beacons, labels, and energy pulses remain an overlay driven only
// by the recorded run artifact.

const SECTIONS = 8;
const OVERLAY = {
  cyan: 0x1fb6a8,
  amber: 0xff9d2e,
  red: 0xe6482e,
};

const POLE_HEIGHT = 4.2;
const OUTER_ATTACH_Y = 4.25;
const CENTER_ATTACH_Y = 4.68;
const TAKEOFF_ATTACH_Y = 4.3;
const LATERALS = [-0.75, 0, 0.75];

let renderer = null;
let scene = null;
let camera = null;
let canvasElement = null;
let sectionUnits = [];
let statsBySection = [];
let thresholdValue = 1;
let thdLimitValue = 0.05;
let decisionBeacon = null;
let nnLabelSprite = null;
let lastTick = 0;
let labelSprites = [];
let isVisible = true;
let needsRender = true;
let frameQueued = false;
let lastA11yKey = "";

// --- Live telemetry binding ---
let faultSection = -1;
let currentMseRatio = 0;
let currentThdRatio = 0;
let liveFaultActive = false;

// --- Raycasting ---
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let interactiveObjects = [];

// --- Fault visualization materials & lights ---
let faultConductorMaterial = null;
let towerPointLight = null;
let hologramLayers = [];

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

const orbit = {
  azimuth: 0.85,
  polar: 1.05,
  radius: 34,
  target: new THREE.Vector3(0, 2.5, 0),
  dragging: false,
  lastX: 0,
  lastY: 0,
  get auto() {
    return !reducedMotion.matches;
  },
};

function requestRender() {
  needsRender = true;
  if (!frameQueued) {
    frameQueued = true;
    requestAnimationFrame(() => {
      frameQueued = false;
      frame();
    });
  }
}

// Deterministic PRNG so scattered props (trees, bushes, stones) land in the
// same spots after a WebGL context restore rebuilds the scene.
function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Procedural textures
// ---------------------------------------------------------------------------

function makeCanvas(width, height) {
  const surface = document.createElement("canvas");
  surface.width = width;
  surface.height = height;
  return surface;
}

function asColorTexture(surface, repeatX = 1, repeatY = 1) {
  const texture = new THREE.CanvasTexture(surface);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeatX, repeatY);
  return texture;
}

function asDataTexture(surface, repeatX = 1, repeatY = 1) {
  // Bump / roughness style maps stay in linear space.
  const texture = new THREE.CanvasTexture(surface);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeatX, repeatY);
  return texture;
}

function grassTexture() {
  const surface = makeCanvas(512, 512);
  const context = surface.getContext("2d");
  context.fillStyle = "#6e8752";
  context.fillRect(0, 0, 512, 512);
  const patchShades = ["#5c7845", "#7d9a60", "#66804d", "#87a168", "#4f6a3c"];
  for (let patch = 0; patch < 70; patch += 1) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const radius = 26 + Math.random() * 70;
    const gradient = context.createRadialGradient(x, y, 2, x, y, radius);
    const shade = patchShades[patch % patchShades.length];
    gradient.addColorStop(0, `${shade}2e`);
    gradient.addColorStop(1, `${shade}00`);
    context.fillStyle = gradient;
    context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  for (let blade = 0; blade < 5200; blade += 1) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const length = 2 + Math.random() * 4;
    const lean = (Math.random() - 0.5) * 2.4;
    const shade = 90 + Math.random() * 70;
    context.strokeStyle = `rgba(${shade * 0.62}, ${shade}, ${shade * 0.5}, 0.35)`;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + lean, y - length);
    context.stroke();
  }
  return asColorTexture(surface, 16, 16);
}

function groundBumpTexture() {
  const surface = makeCanvas(256, 256);
  const context = surface.getContext("2d");
  context.fillStyle = "#7f7f7f";
  context.fillRect(0, 0, 256, 256);
  for (let grain = 0; grain < 5200; grain += 1) {
    const shade = 96 + Math.floor(Math.random() * 78);
    context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
    const size = 1 + Math.random() * 2.4;
    context.fillRect(Math.random() * 256, Math.random() * 256, size, size);
  }
  return asDataTexture(surface, 24, 24);
}

function gravelTexture() {
  const surface = makeCanvas(512, 512);
  const context = surface.getContext("2d");
  context.fillStyle = "#96938b";
  context.fillRect(0, 0, 512, 512);
  const stoneShades = [
    "#8b8882", "#a4a19a", "#7d7a74", "#b1aea7", "#6f6d68", "#9d968a",
  ];
  for (let stone = 0; stone < 1050; stone += 1) {
    const x = Math.random() * 512;
    const y = Math.random() * 512;
    const rx = 2.2 + Math.random() * 4.6;
    const ry = rx * (0.6 + Math.random() * 0.5);
    const angle = Math.random() * Math.PI;
    // Shadow crescent below each stone, then the lit stone body above it,
    // reads as loose crushed rock under a high sun.
    context.fillStyle = "rgba(52, 50, 46, 0.4)";
    context.beginPath();
    context.ellipse(x + 0.9, y + 1.1, rx, ry, angle, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = stoneShades[stone % stoneShades.length];
    context.beginPath();
    context.ellipse(x, y, rx, ry, angle, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "rgba(255, 255, 250, 0.16)";
    context.beginPath();
    context.ellipse(x - rx * 0.25, y - ry * 0.3, rx * 0.45, ry * 0.4, angle, 0, Math.PI * 2);
    context.fill();
  }
  return asColorTexture(surface, 3.4, 3.4);
}

function dirtRoadTexture() {
  const surface = makeCanvas(256, 512);
  const context = surface.getContext("2d");
  context.fillStyle = "#8d7f66";
  context.fillRect(0, 0, 256, 512);
  for (let grain = 0; grain < 2600; grain += 1) {
    const shade = 105 + Math.random() * 60;
    context.fillStyle =
      `rgba(${shade}, ${shade * 0.88}, ${shade * 0.68}, 0.5)`;
    const size = 1 + Math.random() * 2;
    context.fillRect(Math.random() * 256, Math.random() * 512, size, size);
  }
  // Twin wheel ruts running the length of the road.
  for (const rutCenter of [74, 182]) {
    const gradient = context.createLinearGradient(rutCenter - 20, 0, rutCenter + 20, 0);
    gradient.addColorStop(0, "rgba(74, 63, 47, 0)");
    gradient.addColorStop(0.5, "rgba(74, 63, 47, 0.42)");
    gradient.addColorStop(1, "rgba(74, 63, 47, 0)");
    context.fillStyle = gradient;
    context.fillRect(rutCenter - 20, 0, 40, 512);
  }
  // Sparse grass invading the crown between the ruts.
  for (let tuft = 0; tuft < 210; tuft += 1) {
    const x = 106 + Math.random() * 44;
    const y = Math.random() * 512;
    context.strokeStyle = "rgba(96, 122, 74, 0.5)";
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + (Math.random() - 0.5) * 3, y - 2 - Math.random() * 3);
    context.stroke();
  }
  return asColorTexture(surface, 1, 7);
}

function woodTexture() {
  const surface = makeCanvas(128, 256);
  const context = surface.getContext("2d");
  context.fillStyle = "#6b4f36";
  context.fillRect(0, 0, 128, 256);
  for (let streak = 0; streak < 70; streak += 1) {
    const shade = 58 + Math.random() * 58;
    context.strokeStyle =
      `rgba(${shade * 0.92}, ${shade * 0.62}, ${shade * 0.4}, 0.55)`;
    context.lineWidth = 1 + Math.random() * 2;
    const x = Math.random() * 128;
    context.beginPath();
    context.moveTo(x, 0);
    context.bezierCurveTo(
      x + (Math.random() - 0.5) * 8, 85,
      x + (Math.random() - 0.5) * 8, 170,
      x + (Math.random() - 0.5) * 6, 256,
    );
    context.stroke();
  }
  // Weathered silver-gray checks near the surface.
  for (let check = 0; check < 26; check += 1) {
    context.strokeStyle = "rgba(190, 182, 168, 0.18)";
    context.lineWidth = 1;
    const x = Math.random() * 128;
    const y = Math.random() * 256;
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + (Math.random() - 0.5) * 3, y + 14 + Math.random() * 26);
    context.stroke();
  }
  return asColorTexture(surface);
}

function galvanizedTexture() {
  const surface = makeCanvas(256, 256);
  const context = surface.getContext("2d");
  context.fillStyle = "#a7aeb3";
  context.fillRect(0, 0, 256, 256);
  // Spangle patches typical of hot-dip galvanizing.
  const spangles = ["#9ba3a9", "#b3bbc0", "#a0a8ad", "#adb5ba", "#949ca2"];
  for (let patch = 0; patch < 420; patch += 1) {
    context.fillStyle = spangles[patch % spangles.length];
    context.save();
    context.translate(Math.random() * 256, Math.random() * 256);
    context.rotate(Math.random() * Math.PI);
    context.globalAlpha = 0.5;
    const size = 4 + Math.random() * 12;
    context.beginPath();
    context.moveTo(-size / 2, 0);
    context.lineTo(0, -size / 2);
    context.lineTo(size / 2, 0);
    context.lineTo(0, size / 2);
    context.closePath();
    context.fill();
    context.restore();
  }
  context.globalAlpha = 1;
  for (let streak = 0; streak < 40; streak += 1) {
    context.strokeStyle = "rgba(120, 128, 134, 0.12)";
    context.lineWidth = 1 + Math.random();
    const x = Math.random() * 256;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x + (Math.random() - 0.5) * 10, 256);
    context.stroke();
  }
  return asColorTexture(surface, 1.6, 1.6);
}

function tankPaintTexture() {
  const surface = makeCanvas(256, 256);
  const context = surface.getContext("2d");
  context.fillStyle = "#5d6e63";
  context.fillRect(0, 0, 256, 256);
  for (let scuff = 0; scuff < 900; scuff += 1) {
    const shade = 84 + Math.random() * 34;
    context.fillStyle =
      `rgba(${shade}, ${shade + 12}, ${shade + 2}, 0.24)`;
    const size = 1 + Math.random() * 3;
    context.fillRect(Math.random() * 256, Math.random() * 256, size, size);
  }
  // Rust wash bleeding down from lid seams and bolt lines.
  for (let streak = 0; streak < 26; streak += 1) {
    const x = Math.random() * 256;
    const top = Math.random() * 40;
    const length = 40 + Math.random() * 140;
    const gradient = context.createLinearGradient(0, top, 0, top + length);
    gradient.addColorStop(0, "rgba(96, 62, 34, 0.28)");
    gradient.addColorStop(1, "rgba(96, 62, 34, 0)");
    context.fillStyle = gradient;
    context.fillRect(x, top, 1.6 + Math.random() * 2.4, length);
  }
  // Horizontal weld bands.
  for (const band of [86, 170]) {
    context.fillStyle = "rgba(38, 46, 41, 0.35)";
    context.fillRect(0, band, 256, 3);
    context.fillStyle = "rgba(210, 220, 214, 0.12)";
    context.fillRect(0, band + 3, 256, 1.5);
  }
  return asColorTexture(surface);
}

function tankRibBumpTexture() {
  // Vertical corrugation stripes; lit as ribs by the bump channel.
  const surface = makeCanvas(256, 256);
  const context = surface.getContext("2d");
  context.fillStyle = "#808080";
  context.fillRect(0, 0, 256, 256);
  for (let x = 0; x < 256; x += 16) {
    const gradient = context.createLinearGradient(x, 0, x + 16, 0);
    gradient.addColorStop(0, "#6a6a6a");
    gradient.addColorStop(0.45, "#a8a8a8");
    gradient.addColorStop(0.55, "#a8a8a8");
    gradient.addColorStop(1, "#5f5f5f");
    context.fillStyle = gradient;
    context.fillRect(x, 0, 16, 256);
  }
  for (const band of [86, 170]) {
    context.fillStyle = "#4c4c4c";
    context.fillRect(0, band, 256, 4);
  }
  return asDataTexture(surface, 2, 1);
}

function concreteTexture() {
  const surface = makeCanvas(256, 256);
  const context = surface.getContext("2d");
  context.fillStyle = "#b7bab4";
  context.fillRect(0, 0, 256, 256);
  for (let grain = 0; grain < 2400; grain += 1) {
    const shade = 150 + Math.random() * 60;
    context.fillStyle = `rgba(${shade}, ${shade}, ${shade - 6}, 0.35)`;
    const size = 1 + Math.random() * 2;
    context.fillRect(Math.random() * 256, Math.random() * 256, size, size);
  }
  for (let stain = 0; stain < 9; stain += 1) {
    const x = Math.random() * 256;
    const y = Math.random() * 256;
    const radius = 14 + Math.random() * 40;
    const gradient = context.createRadialGradient(x, y, 2, x, y, radius);
    gradient.addColorStop(0, "rgba(96, 99, 92, 0.22)");
    gradient.addColorStop(1, "rgba(96, 99, 92, 0)");
    context.fillStyle = gradient;
    context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }
  // Faint horizontal formwork lines.
  for (const line of [64, 128, 192]) {
    context.fillStyle = "rgba(90, 92, 88, 0.16)";
    context.fillRect(0, line, 256, 1.4);
  }
  return asColorTexture(surface, 1.4, 1.4);
}

function chainLinkTexture() {
  const surface = makeCanvas(128, 128);
  const context = surface.getContext("2d");
  context.clearRect(0, 0, 128, 128);
  context.strokeStyle = "rgba(150, 155, 158, 0.9)";
  context.lineWidth = 1.6;
  for (let offset = -128; offset < 256; offset += 12) {
    context.beginPath();
    context.moveTo(offset, 0);
    context.lineTo(offset + 128, 128);
    context.stroke();
    context.beginPath();
    context.moveTo(offset + 128, 0);
    context.lineTo(offset, 128);
    context.stroke();
  }
  const texture = asColorTexture(surface, 6, 1.4);
  return texture;
}

function skyTexture() {
  const surface = makeCanvas(1024, 512);
  const context = surface.getContext("2d");
  const gradient = context.createLinearGradient(0, 0, 0, 512);
  gradient.addColorStop(0, "#4d87c7");
  gradient.addColorStop(0.4, "#7fadd9");
  gradient.addColorStop(0.62, "#b9d4ea");
  gradient.addColorStop(0.74, "#e6eff6");
  gradient.addColorStop(1, "#dcebf5");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 1024, 512);
  // Warm haze near the sun's quadrant.
  const sunGlow = context.createRadialGradient(700, 200, 8, 700, 200, 190);
  sunGlow.addColorStop(0, "rgba(255, 244, 214, 0.85)");
  sunGlow.addColorStop(0.25, "rgba(255, 240, 208, 0.32)");
  sunGlow.addColorStop(1, "rgba(255, 240, 208, 0)");
  context.fillStyle = sunGlow;
  context.fillRect(0, 0, 1024, 512);
  // Fair-weather cumulus: clusters of soft overlapping ellipses.
  for (let cloud = 0; cloud < 13; cloud += 1) {
    const baseX = Math.random() * 1024;
    const baseY = 90 + Math.random() * 190;
    const puffs = 5 + Math.floor(Math.random() * 5);
    for (let puff = 0; puff < puffs; puff += 1) {
      const px = baseX + (Math.random() - 0.5) * 130;
      const py = baseY + (Math.random() - 0.5) * 26;
      const rx = 26 + Math.random() * 56;
      const ry = 9 + Math.random() * 16;
      const soft = context.createRadialGradient(px, py, 1, px, py, rx);
      soft.addColorStop(0, "rgba(255, 255, 255, 0.34)");
      soft.addColorStop(0.7, "rgba(255, 255, 255, 0.14)");
      soft.addColorStop(1, "rgba(255, 255, 255, 0)");
      context.fillStyle = soft;
      context.save();
      context.translate(px, py);
      context.scale(1, ry / rx);
      context.beginPath();
      context.arc(0, 0, rx, 0, Math.PI * 2);
      context.fill();
      context.restore();
    }
  }
  const texture = new THREE.CanvasTexture(surface);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function contactShadowTexture() {
  const surface = makeCanvas(128, 128);
  const context = surface.getContext("2d");
  const gradient = context.createRadialGradient(64, 64, 4, 64, 64, 62);
  gradient.addColorStop(0, "rgba(20, 22, 20, 0.5)");
  gradient.addColorStop(0.65, "rgba(20, 22, 20, 0.24)");
  gradient.addColorStop(1, "rgba(20, 22, 20, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(surface);
  return texture;
}

// ---------------------------------------------------------------------------
// Materials (module-level so a context-restore rebuild reuses them)
// ---------------------------------------------------------------------------

const woodMaterial = new THREE.MeshStandardMaterial({
  map: woodTexture(), roughness: 0.85, metalness: 0.0,
});
const concreteMaterial = new THREE.MeshStandardMaterial({
  map: concreteTexture(), roughness: 0.92, metalness: 0.02,
});
const galvanizedMaterial = new THREE.MeshStandardMaterial({
  map: galvanizedTexture(), color: 0xdfe3e6, roughness: 0.46, metalness: 0.8,
});
const steelDarkMaterial = new THREE.MeshStandardMaterial({
  color: 0x4d5459, roughness: 0.55, metalness: 0.7,
});
const aluminumMaterial = new THREE.MeshStandardMaterial({
  color: 0xd7dde1, roughness: 0.3, metalness: 0.9,
});
const porcelainBrownMaterial = new THREE.MeshStandardMaterial({
  color: 0x6e4634, roughness: 0.16, metalness: 0.0,
});
const porcelainGrayMaterial = new THREE.MeshStandardMaterial({
  color: 0x9aa1a6, roughness: 0.18, metalness: 0.0,
});
const tankMaterial = new THREE.MeshStandardMaterial({
  map: tankPaintTexture(), bumpMap: tankRibBumpTexture(), bumpScale: 1.6,
  roughness: 0.58, metalness: 0.34,
});
const radiatorMaterial = new THREE.MeshStandardMaterial({
  color: 0x54655b, roughness: 0.62, metalness: 0.32,
});
const conductorMaterial = new THREE.MeshStandardMaterial({
  color: 0x3a3d40, roughness: 0.42, metalness: 0.82,
});
const copperMaterial = new THREE.MeshStandardMaterial({
  color: 0x9a5c33, roughness: 0.4, metalness: 0.85,
});
const canopyMaterial = new THREE.MeshStandardMaterial({
  color: 0x47663a, roughness: 0.95, metalness: 0.0, flatShading: true,
});
const contactShadowMaterial = new THREE.MeshBasicMaterial({
  map: contactShadowTexture(), transparent: true, depthWrite: false,
});

faultConductorMaterial = new THREE.MeshStandardMaterial({
  color: 0xe6482e,
  emissive: 0xff3311,
  emissiveIntensity: 0.6,
  roughness: 0.35,
  metalness: 0.85,
});

// ---------------------------------------------------------------------------
// Shared geometries (built once, reused by every pole / string / tree)
// ---------------------------------------------------------------------------

function lathePoints(pairs) {
  return pairs.map(([radius, y]) => new THREE.Vector2(radius, y));
}

// Pin-type distribution insulator: three glossy porcelain sheds on a core.
const pinInsulatorGeometry = new THREE.LatheGeometry(lathePoints([
  [0.004, 0], [0.078, 0.006], [0.094, 0.034], [0.052, 0.062],
  [0.052, 0.096], [0.104, 0.122], [0.086, 0.152], [0.05, 0.176],
  [0.05, 0.206], [0.082, 0.23], [0.066, 0.258], [0.034, 0.282],
  [0.004, 0.298],
]), 20);

// Suspension string: a rod of cap-and-pin discs, built as one lathe profile
// hanging downward from its attachment origin.
function discStringGeometry(discs) {
  const pairs = [[0.02, 0]];
  let y = -0.05;
  for (let disc = 0; disc < discs; disc += 1) {
    pairs.push([0.024, y]);
    pairs.push([0.105, y - 0.028]);
    pairs.push([0.115, y - 0.05]);
    pairs.push([0.03, y - 0.082]);
    y -= 0.1;
  }
  pairs.push([0.02, y]);
  pairs.push([0.004, y - 0.03]);
  return new THREE.LatheGeometry(lathePoints(pairs), 16);
}
const takeoffStringGeometry = discStringGeometry(4);

// Transformer high-voltage bushing: tall shed stack with a metal cap.
const hvBushingGeometry = new THREE.LatheGeometry(lathePoints([
  [0.004, 0], [0.075, 0.004], [0.075, 0.05],
  [0.15, 0.085], [0.06, 0.13], [0.15, 0.165], [0.06, 0.21],
  [0.14, 0.245], [0.058, 0.29], [0.135, 0.325], [0.055, 0.37],
  [0.125, 0.405], [0.052, 0.45], [0.115, 0.485], [0.05, 0.53],
  [0.1, 0.565], [0.048, 0.61], [0.07, 0.64], [0.05, 0.7], [0.004, 0.71],
]), 20);

const pulseGeometry = new THREE.SphereGeometry(0.16, 10, 10);
const trunkTreeGeometry = new THREE.CylinderGeometry(0.1, 0.16, 1.2, 7);
const canopyGeometry = new THREE.IcosahedronGeometry(1.2, 1);
const bushGeometry = new THREE.IcosahedronGeometry(0.4, 1);
const boltGeometry = new THREE.CylinderGeometry(0.026, 0.026, 0.06, 6);

function makeLabelSprite(title, subtitle, accent = "#dce8ea", track = false) {
  const surface = document.createElement("canvas");
  surface.width = 512;
  surface.height = 128;
  const context = surface.getContext("2d");
  context.fillStyle = "rgba(20, 30, 34, 0.66)";
  context.fillRect(0, 0, 512, 128);
  context.fillStyle = "#f4fafb";
  context.font = "700 34px 'Cascadia Code', ui-monospace, monospace";
  context.fillText(title, 18, 52);
  context.fillStyle = accent;
  context.font = "26px 'Cascadia Code', ui-monospace, monospace";
  context.fillText(subtitle, 18, 96);
  const texture = new THREE.CanvasTexture(surface);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, transparent: true }),
  );
  // Base scale is stored so resize() can grow labels on narrow viewports.
  sprite.userData.baseScale = [5.2, 1.3];
  sprite.scale.set(5.2, 1.3, 1);
  if (track) labelSprites.push(sprite);
  return sprite;
}

function retextureSprite(sprite, title, subtitle, accent) {
  const replacement = makeLabelSprite(title, subtitle, accent);
  sprite.material.map.dispose();
  sprite.material.dispose();
  sprite.material = replacement.material;
}

// ---------------------------------------------------------------------------
// Conductors: true catenary sag rendered as a shaded tube
// ---------------------------------------------------------------------------

function catenaryCurve(from, to, sag) {
  const points = [];
  const stiffness = 2.4;
  const edge = Math.cosh(stiffness * 0.5);
  for (let step = 0; step <= 22; step += 1) {
    const t = step / 22;
    const point = from.clone().lerp(to, t);
    // Normalized catenary dip: 0 at both supports, 1 at mid-span.
    const dip = (edge - Math.cosh(stiffness * (t - 0.5))) / (edge - 1);
    point.y -= sag * dip;
    points.push(point);
  }
  return new THREE.CatmullRomCurve3(points);
}

function conductorSpan(from, to, sagOverride = null) {
  const length = from.distanceTo(to);
  const sag = sagOverride ?? Math.min(1.1, 0.032 * length * length);
  const curve = catenaryCurve(from, to, sag);
  const mesh = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 24, 0.022, 5, false),
    conductorMaterial,
  );
  mesh.castShadow = true;
  return { mesh, curve };
}

function shadowed(mesh) {
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

function contactShadow(width, depth) {
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth), contactShadowMaterial,
  );
  plane.rotation.x = -Math.PI / 2;
  plane.renderOrder = 1;
  return plane;
}

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

function makePole(height = POLE_HEIGHT) {
  const pole = new THREE.Group();
  const footing = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.2, 0.26, 0.3, 10), concreteMaterial,
  ));
  footing.position.y = 0.15;
  pole.add(footing);

  const trunk = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.085, 0.145, height, 10), woodMaterial,
  ));
  trunk.position.y = height / 2;
  pole.add(trunk);
  const cap = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.098, 0.09, 0.05, 10), galvanizedMaterial,
  ));
  cap.position.y = height + 0.02;
  pole.add(cap);

  // Wood crossarm with a visible through-bolt and steel V-braces.
  const armY = height - 0.42;
  const crossarm = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(2.3, 0.14, 0.11), woodMaterial,
  ));
  crossarm.position.y = armY;
  pole.add(crossarm);
  const throughBolt = shadowed(new THREE.Mesh(boltGeometry, galvanizedMaterial));
  throughBolt.rotation.x = Math.PI / 2;
  throughBolt.scale.set(1, 4.6, 1);
  throughBolt.position.set(0, armY, 0);
  pole.add(throughBolt);
  for (const side of [-1, 1]) {
    const brace = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.92, 0.022), galvanizedMaterial,
    ));
    brace.position.set(side * 0.44, armY - 0.4, 0.08);
    brace.rotation.z = side * 0.62;
    pole.add(brace);
  }

  // Outer phases on crossarm pins, center phase on a pole-top pin: the
  // classic three-phase distribution arrangement.
  for (const offset of [-0.75, 0.75]) {
    const pin = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.024, 0.16, 8), galvanizedMaterial,
    ));
    pin.position.set(offset, armY + 0.14, 0);
    pole.add(pin);
    const insulator = shadowed(new THREE.Mesh(
      pinInsulatorGeometry, porcelainBrownMaterial,
    ));
    insulator.position.set(offset, armY + 0.17, 0);
    pole.add(insulator);
  }
  const topPin = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.02, 0.026, 0.2, 8), galvanizedMaterial,
  ));
  topPin.position.y = height + 0.12;
  pole.add(topPin);
  const topInsulator = shadowed(new THREE.Mesh(
    pinInsulatorGeometry, porcelainBrownMaterial,
  ));
  topInsulator.position.y = height + 0.18;
  pole.add(topInsulator);
  return pole;
}

function makeTakeoffStructure() {
  // Steel riser mast just inside the fence where each feeder leaves the
  // substation: tubular post, steel arm, suspension strings on every phase.
  const takeoff = new THREE.Group();
  const basePlate = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(0.5, 0.06, 0.5), galvanizedMaterial,
  ));
  basePlate.position.y = 0.03;
  takeoff.add(basePlate);
  const post = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.085, 0.11, 4.95, 10), galvanizedMaterial,
  ));
  post.position.y = 2.48;
  takeoff.add(post);
  const arm = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(2.1, 0.12, 0.12), galvanizedMaterial,
  ));
  arm.position.y = 4.78;
  takeoff.add(arm);
  for (const side of [-1, 1]) {
    const knee = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.68, 0.03), galvanizedMaterial,
    ));
    knee.position.set(side * 0.42, 4.52, 0);
    knee.rotation.z = side * 0.75;
    takeoff.add(knee);
  }
  for (const offset of LATERALS) {
    const string = shadowed(new THREE.Mesh(
      takeoffStringGeometry, porcelainGrayMaterial,
    ));
    string.position.set(offset, 4.72, 0);
    takeoff.add(string);
  }
  return takeoff;
}

function makeTransformer() {
  const unit = new THREE.Group();

  const plinth = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(2.7, 0.24, 2.0), concreteMaterial,
  ));
  plinth.position.y = 0.12;
  unit.add(plinth);
  const shadow = contactShadow(3.4, 2.7);
  shadow.position.y = 0.02;
  unit.add(shadow);
  for (const side of [-1, 1]) {
    const rail = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(2.1, 0.09, 0.14), steelDarkMaterial,
    ));
    rail.position.set(0, 0.28, side * 0.48);
    unit.add(rail);
  }

  // Ribbed main tank (bump-mapped corrugation) with corner stiffeners.
  const tankHeight = 1.55;
  const tankY = 0.33 + tankHeight / 2;
  const tank = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(1.8, tankHeight, 1.2), tankMaterial,
  ));
  tank.position.y = tankY;
  unit.add(tank);
  for (const cornerX of [-0.88, 0.88]) {
    for (const cornerZ of [-0.58, 0.58]) {
      const stiffener = shadowed(new THREE.Mesh(
        new THREE.BoxGeometry(0.07, tankHeight, 0.07), steelDarkMaterial,
      ));
      stiffener.position.set(cornerX, tankY, cornerZ);
      unit.add(stiffener);
    }
  }
  for (const lugX of [-0.7, 0.7]) {
    const lug = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(0.14, 0.1, 0.05), steelDarkMaterial,
    ));
    lug.position.set(lugX, tankY + tankHeight / 2 - 0.1, 0.62);
    unit.add(lug);
  }

  // Bolted lid flange.
  const lidY = 0.33 + tankHeight + 0.05;
  const lid = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(1.92, 0.1, 1.32), steelDarkMaterial,
  ));
  lid.position.y = lidY;
  unit.add(lid);
  for (let bolt = 0; bolt < 10; bolt += 1) {
    const boltMesh = new THREE.Mesh(boltGeometry, galvanizedMaterial);
    const alongFront = bolt < 5;
    const boltX = -0.8 + (bolt % 5) * 0.4;
    boltMesh.position.set(boltX, lidY + 0.06, alongFront ? 0.6 : -0.6);
    unit.add(boltMesh);
  }

  // Radiator banks: vertical fin panels tied into header pipes, standing
  // off both long sides of the tank.
  for (const side of [-1, 1]) {
    const bank = new THREE.Group();
    for (const header of [0.62, 1.72]) {
      const pipe = shadowed(new THREE.Mesh(
        new THREE.CylinderGeometry(0.05, 0.05, 1.34, 8), radiatorMaterial,
      ));
      pipe.rotation.x = Math.PI / 2;
      pipe.position.set(0, header, 0);
      bank.add(pipe);
      const stub = shadowed(new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.045, 0.3, 8), radiatorMaterial,
      ));
      stub.rotation.z = Math.PI / 2;
      stub.position.set(side * -0.18, header, 0);
      bank.add(stub);
    }
    for (let panel = 0; panel < 7; panel += 1) {
      const fin = shadowed(new THREE.Mesh(
        new THREE.BoxGeometry(0.3, 1.06, 0.042), radiatorMaterial,
      ));
      fin.position.set(0.02, 1.17, -0.6 + panel * 0.2);
      bank.add(fin);
    }
    bank.position.set(side * 1.24, 0, 0);
    unit.add(bank);
  }

  // Conservator drum above the lid with support legs and an oil gauge.
  const conservator = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.21, 0.21, 1.05, 14), tankMaterial,
  ));
  conservator.rotation.z = Math.PI / 2;
  conservator.position.set(0.28, lidY + 0.52, -0.34);
  unit.add(conservator);
  for (const legX of [-0.1, 0.66]) {
    const leg = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.34, 0.05), steelDarkMaterial,
    ));
    leg.position.set(legX, lidY + 0.22, -0.34);
    unit.add(leg);
  }
  const gauge = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.055, 0.03, 10),
    new THREE.MeshStandardMaterial({ color: 0xe9e7de, roughness: 0.3 }),
  );
  gauge.rotation.z = Math.PI / 2;
  gauge.position.set(0.82, lidY + 0.52, -0.34);
  unit.add(gauge);
  const drop = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.035, 0.035, 0.42, 8), steelDarkMaterial,
  ));
  drop.position.set(-0.2, lidY + 0.26, -0.34);
  unit.add(drop);

  // Three HV bushings on lid turrets, with metal caps and stud terminals.
  for (const offset of [-0.55, 0, 0.55]) {
    const turret = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.12, 0.16, 10), steelDarkMaterial,
    ));
    turret.position.set(offset, lidY + 0.12, 0.16);
    unit.add(turret);
    const bushing = shadowed(new THREE.Mesh(
      hvBushingGeometry, porcelainGrayMaterial,
    ));
    bushing.position.set(offset, lidY + 0.2, 0.16);
    unit.add(bushing);
    const capTop = new THREE.Mesh(
      new THREE.CylinderGeometry(0.045, 0.055, 0.07, 8), aluminumMaterial,
    );
    capTop.position.set(offset, lidY + 0.94, 0.16);
    unit.add(capTop);
  }

  // Low-voltage bushings on the rear face.
  for (const offset of [-0.45, 0, 0.45]) {
    const lv = shadowed(new THREE.Mesh(
      pinInsulatorGeometry, porcelainGrayMaterial,
    ));
    lv.scale.setScalar(0.85);
    lv.rotation.x = -Math.PI / 2;
    lv.position.set(offset, tankY + 0.42, -0.62);
    unit.add(lv);
  }

  // Drain valve with a handwheel, nameplate, and ground strap.
  const valveStem = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.03, 0.03, 0.16, 8), steelDarkMaterial,
  ));
  valveStem.rotation.x = Math.PI / 2;
  valveStem.position.set(-0.62, 0.52, 0.64);
  unit.add(valveStem);
  const handwheel = new THREE.Mesh(
    new THREE.TorusGeometry(0.07, 0.017, 6, 14),
    new THREE.MeshStandardMaterial({ color: 0x8a2f24, roughness: 0.5 }),
  );
  handwheel.position.set(-0.62, 0.52, 0.74);
  unit.add(handwheel);
  const nameplate = new THREE.Mesh(
    new THREE.PlaneGeometry(0.3, 0.38),
    new THREE.MeshStandardMaterial({ color: 0xd8d5c9, roughness: 0.35, metalness: 0.2 }),
  );
  nameplate.position.set(0.42, tankY, 0.615);
  unit.add(nameplate);
  const strap = new THREE.Mesh(
    new THREE.BoxGeometry(0.025, 0.9, 0.012), copperMaterial,
  );
  strap.position.set(-0.86, 0.72, 0.6);
  unit.add(strap);

  return unit;
}

function makeGantry(width = 7) {
  const gantry = new THREE.Group();
  const legHeight = 4.4;
  for (const side of [-1, 1]) {
    const plate = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(0.56, 0.08, 0.56), galvanizedMaterial,
    ));
    plate.position.set(side * (width / 2), 0.04, 0);
    gantry.add(plate);
    const post = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.11, 0.15, legHeight, 10), galvanizedMaterial,
    ));
    post.position.set(side * (width / 2), legHeight / 2, 0);
    gantry.add(post);
  }
  // Box-truss beam: two chords with zig-zag lattice diagonals.
  for (const chordY of [4.2, 4.52]) {
    const chord = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.12, 0.12), galvanizedMaterial,
    ));
    chord.position.y = chordY;
    gantry.add(chord);
  }
  const bays = 8;
  for (let bay = 0; bay < bays; bay += 1) {
    const x0 = -width / 2 + (bay * width) / bays;
    const diagonal = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.028, 0.028, Math.hypot(width / bays, 0.32), 6),
      galvanizedMaterial,
    ));
    diagonal.position.set(x0 + width / bays / 2, 4.36, 0);
    diagonal.rotation.z = (bay % 2 ? 1 : -1) *
      Math.atan2(width / bays, 0.32);
    gantry.add(diagonal);
  }
  return gantry;
}

function makeFence(size = 13, height = 2.1) {
  const fence = new THREE.Group();
  const mesh = chainLinkTexture();
  const material = new THREE.MeshStandardMaterial({
    map: mesh, transparent: true, side: THREE.DoubleSide,
    roughness: 0.5, metalness: 0.6, alphaTest: 0.15,
  });
  const half = size / 2;
  const sides = [
    { position: [0, height / 2, -half], rotation: 0 },
    { position: [0, height / 2, half], rotation: 0 },
    { position: [-half, height / 2, 0], rotation: Math.PI / 2 },
    { position: [half, height / 2, 0], rotation: Math.PI / 2 },
  ];
  for (const side of sides) {
    const panel = new THREE.Mesh(
      new THREE.PlaneGeometry(size, height), material,
    );
    panel.position.set(...side.position);
    panel.rotation.y = side.rotation;
    fence.add(panel);
    // Top rail along every run.
    const rail = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.03, 0.03, size, 8), galvanizedMaterial,
    ));
    rail.rotation.z = Math.PI / 2;
    rail.position.set(...side.position);
    rail.position.y = height;
    rail.rotation.y = side.rotation;
    fence.add(rail);
    // Three strands of barbed wire on outward-leaning extension arms.
    for (let strand = 0; strand < 3; strand += 1) {
      const wire = new THREE.Mesh(
        new THREE.CylinderGeometry(0.008, 0.008, size, 5), steelDarkMaterial,
      );
      wire.rotation.z = Math.PI / 2;
      wire.position.set(...side.position);
      wire.rotation.y = side.rotation;
      const outward = 0.06 + strand * 0.07;
      wire.position.y = height + 0.12 + strand * 0.1;
      if (side.rotation === 0) {
        wire.position.z += Math.sign(side.position[2]) * outward;
      } else {
        wire.position.x += Math.sign(side.position[0]) * outward;
      }
      fence.add(wire);
    }
  }
  const postSpots = [];
  for (const x of [-half, 0, half]) {
    for (const z of [-half, 0, half]) {
      if (x === 0 && z === 0) continue;
      postSpots.push([x, z]);
    }
  }
  for (const [x, z] of postSpots) {
    const post = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, height, 8), galvanizedMaterial,
    ));
    post.position.set(x, height / 2, z);
    fence.add(post);
    const barbArm = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.42, 6), galvanizedMaterial,
    ));
    barbArm.position.set(x, height + 0.18, z);
    const lean = new THREE.Vector3(x, 0, z).normalize().multiplyScalar(0.32);
    barbArm.rotation.set(lean.z, 0, -lean.x);
    fence.add(barbArm);
  }
  // Vehicle gate on the +x side, where the access road meets the yard.
  for (const gateHalf of [-1, 1]) {
    const gateLeaf = shadowed(new THREE.Mesh(
      new THREE.BoxGeometry(0.06, height - 0.24, 1.55), galvanizedMaterial,
    ));
    gateLeaf.position.set(half + 0.09, (height - 0.24) / 2 + 0.1, gateHalf * 0.85);
    fence.add(gateLeaf);
  }
  return fence;
}

function makeTree(random) {
  const tree = new THREE.Group();
  const trunk = shadowed(new THREE.Mesh(trunkTreeGeometry, woodMaterial));
  trunk.position.y = 0.6;
  tree.add(trunk);
  const puffs = 2 + Math.floor(random() * 2);
  for (let puff = 0; puff < puffs; puff += 1) {
    const canopy = shadowed(new THREE.Mesh(canopyGeometry, canopyMaterial));
    canopy.position.set(
      (random() - 0.5) * 0.9,
      1.55 + puff * 0.75 + random() * 0.3,
      (random() - 0.5) * 0.9,
    );
    canopy.scale.setScalar(0.75 + random() * 0.5);
    canopy.scale.y *= 0.85;
    tree.add(canopy);
  }
  return tree;
}

function makeControlHouse(houseBase) {
  const house = new THREE.Group();
  const shadow = contactShadow(4.4, 3.7);
  shadow.position.copy(houseBase).y = 0.006;
  house.add(shadow);
  const walls = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(3.1, 2.5, 2.5), concreteMaterial,
  ));
  walls.position.copy(houseBase).y = 1.31;
  house.add(walls);
  const roof = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(3.5, 0.18, 2.9),
    new THREE.MeshStandardMaterial({
      color: 0x4a4f52, roughness: 0.7, metalness: 0.3,
    }),
  ));
  roof.position.copy(houseBase).y = 2.65;
  house.add(roof);
  const doorFrame = new THREE.Mesh(
    new THREE.PlaneGeometry(1.05, 1.92),
    new THREE.MeshStandardMaterial({ color: 0x9aa0a4, roughness: 0.55, metalness: 0.4 }),
  );
  doorFrame.position.copy(houseBase).add(new THREE.Vector3(0, 1.0, 1.253));
  house.add(doorFrame);
  const door = new THREE.Mesh(
    new THREE.PlaneGeometry(0.85, 1.75),
    new THREE.MeshStandardMaterial({
      color: 0x3c454b, roughness: 0.6, metalness: 0.4,
    }),
  );
  door.position.copy(houseBase).add(new THREE.Vector3(0, 0.94, 1.257));
  house.add(door);
  const sign = new THREE.Mesh(
    new THREE.PlaneGeometry(0.5, 0.34),
    new THREE.MeshStandardMaterial({ color: 0xe8c437, roughness: 0.5 }),
  );
  sign.position.copy(houseBase).add(new THREE.Vector3(0.95, 1.5, 1.253));
  house.add(sign);
  const hvac = shadowed(new THREE.Mesh(
    new THREE.BoxGeometry(0.8, 0.7, 0.6), galvanizedMaterial,
  ));
  hvac.position.copy(houseBase).add(new THREE.Vector3(1.95, 0.36, 0.4));
  house.add(hvac);
  // Service mast that receives the overhead feed from the yard.
  const mast = shadowed(new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 1.1, 8), galvanizedMaterial,
  ));
  mast.position.copy(houseBase).add(new THREE.Vector3(-1.2, 3.2, 0.6));
  house.add(mast);
  const conduit = new THREE.Mesh(
    new THREE.CylinderGeometry(0.028, 0.028, 2.5, 8), galvanizedMaterial,
  );
  conduit.position.copy(houseBase).add(new THREE.Vector3(-1.58, 1.35, 0.6));
  house.add(conduit);
  return { house, mastTip: houseBase.clone().add(new THREE.Vector3(-1.2, 3.72, 0.6)) };
}

// ---------------------------------------------------------------------------
// Scene assembly
// ---------------------------------------------------------------------------

function disposeSceneGraph() {
  if (!scene) return;
  const disposedTextures = new Set();
  scene.traverse((object) => {
    // Shadow-casting lights own a render target that dispose() must free.
    if (object.isLight) object.dispose?.();
    object.geometry?.dispose?.();
    const materials = Array.isArray(object.material)
      ? object.material
      : object.material
        ? [object.material]
        : [];
    for (const material of materials) {
      for (const value of Object.values(material)) {
        if (value?.isTexture && !disposedTextures.has(value)) {
          disposedTextures.add(value);
          value.dispose();
        }
      }
      material.dispose?.();
    }
  });
}

function buildStaticScene() {
  disposeSceneGraph();
  labelSprites = [];
  sectionUnits = [];
  decisionBeacon = null;
  nnLabelSprite = null;
  interactiveObjects = [];
  hologramLayers = [];
  faultSection = -1;
  currentMseRatio = 0;
  liveFaultActive = false;
  const random = mulberry32(42);
  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0xdfe9f2, 60, 150);

  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(130, 24, 16),
    new THREE.MeshBasicMaterial({
      map: skyTexture(), side: THREE.BackSide, fog: false,
    }),
  );
  scene.add(sky);

  // Lighting tuned for depth: warm low key light for long shadows, cool
  // sky fill from the opposite side, soft hemisphere ambient.
  scene.add(new THREE.HemisphereLight(0xcfe2f7, 0x88976c, 0.75));
  const sun = new THREE.DirectionalLight(0xffe8c4, 2.3);
  sun.position.set(28, 38, 16);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -42;
  sun.shadow.camera.right = 42;
  sun.shadow.camera.top = 42;
  sun.shadow.camera.bottom = -42;
  sun.shadow.camera.far = 130;
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.02;
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x9db8d9, 0.5);
  fill.position.set(-26, 18, -30);
  scene.add(fill);

  const grass = new THREE.Mesh(
    new THREE.CircleGeometry(95, 48),
    new THREE.MeshStandardMaterial({
      map: grassTexture(), bumpMap: groundBumpTexture(), bumpScale: 0.3,
      roughness: 0.95, metalness: 0.0,
    }),
  );
  grass.rotation.x = -Math.PI / 2;
  grass.receiveShadow = true;
  scene.add(grass);

  // Dirt access road from the gate on +x out to the horizon.
  const road = new THREE.Mesh(
    new THREE.PlaneGeometry(3.4, 42),
    new THREE.MeshStandardMaterial({
      map: dirtRoadTexture(), roughness: 0.95, metalness: 0.0,
    }),
  );
  road.rotation.x = -Math.PI / 2;
  road.rotation.z = Math.PI / 2;
  road.position.set(27.4, 0.012, 0);
  road.receiveShadow = true;
  scene.add(road);

  const gravelPad = new THREE.Mesh(
    new THREE.BoxGeometry(13.5, 0.12, 13.5),
    new THREE.MeshStandardMaterial({
      map: gravelTexture(), roughness: 0.95, metalness: 0.02,
    }),
  );
  gravelPad.position.y = 0.06;
  gravelPad.receiveShadow = true;
  scene.add(gravelPad);
  scene.add(makeFence());

  const transformerA = makeTransformer();
  transformerA.position.set(-2.6, 0.12, 2.6);
  scene.add(transformerA);
  const transformerB = makeTransformer();
  transformerB.position.set(2.6, 0.12, 2.6);
  transformerB.rotation.y = Math.PI;
  scene.add(transformerB);

  const gantryFront = makeGantry();
  gantryFront.position.set(0, 0.12, -3.4);
  scene.add(gantryFront);
  const gantryBack = makeGantry();
  gantryBack.position.set(0, 0.12, 3.4);
  scene.add(gantryBack);

  for (const offset of [-1.1, 0, 1.1]) {
    const busbar = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.055, 0.055, 6.8, 12), aluminumMaterial,
    ));
    busbar.rotation.x = Math.PI / 2;
    busbar.position.set(offset, 5.08, 0);
    scene.add(busbar);
    for (const z of [-3.4, 3.4]) {
      // Upright post insulator standing on the gantry top chord. The gantry
      // group is lifted 0.12, so the chord's top face sits at world
      // y = 0.12 + 4.52 + 0.06 = 4.70; the 0.343-tall post carries the bus
      // at its tip (~5.04).
      const post = shadowed(new THREE.Mesh(
        pinInsulatorGeometry, porcelainGrayMaterial,
      ));
      post.scale.setScalar(1.15);
      post.position.set(offset, 4.7, z);
      scene.add(post);
    }
  }
  const substationLabel = makeLabelSprite(
    "SUBSTATION BUS", "three-phase busbar · recorded run source", "#9fc0e8", true,
  );
  substationLabel.position.set(0, 7.4, 0);
  scene.add(substationLabel);

  // Ring of trees and brush outside the yard for horizon depth. The +x
  // corridor stays clear for the access road.
  for (let tree = 0; tree < 16; tree += 1) {
    const angle = random() * Math.PI * 2;
    if (Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle))) < 0.32) continue;
    const distance = 52 + random() * 26;
    const treeMesh = makeTree(random);
    treeMesh.position.set(
      Math.cos(angle) * distance, 0, Math.sin(angle) * distance,
    );
    treeMesh.scale.setScalar(1.3 + random() * 1.4);
    treeMesh.rotation.y = random() * Math.PI * 2;
    scene.add(treeMesh);
  }
  for (let bush = 0; bush < 12; bush += 1) {
    const angle = random() * Math.PI * 2;
    // Keep the +x access-road corridor clear, same as the tree ring.
    if (Math.abs(Math.atan2(Math.sin(angle), Math.cos(angle))) < 0.35) continue;
    const distance = 10.5 + random() * 9;
    const bushMesh = shadowed(new THREE.Mesh(bushGeometry, canopyMaterial));
    bushMesh.position.set(
      Math.cos(angle) * distance, 0.22, Math.sin(angle) * distance,
    );
    bushMesh.scale.set(
      0.8 + random() * 0.9, 0.55 + random() * 0.5, 0.8 + random() * 0.9,
    );
    scene.add(bushMesh);
  }

  sectionUnits = [];
  for (let section = 0; section < SECTIONS; section += 1) {
    const angle = (section / SECTIONS) * Math.PI * 2;
    const direction = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle));
    const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x);
    const unit = { pulses: [], curves: [], conductors: [] };

    const takeoff = makeTakeoffStructure();
    takeoff.position.copy(direction.clone().multiplyScalar(4.6)).setY(0.12);
    takeoff.lookAt(new THREE.Vector3(0, takeoff.position.y, 0));
    scene.add(takeoff);

    const distances = [9, 14, 19];
    const poleBases = [];
    for (const distance of distances) {
      const pole = makePole();
      pole.position.copy(direction.clone().multiplyScalar(distance));
      pole.lookAt(new THREE.Vector3(0, 0, 0));
      scene.add(pole);
      poleBases.push(pole.position.clone());
    }

    // Guy wire and anchor behind the dead-end pole.
    const lastBase = poleBases[2];
    const guyTop = lastBase.clone().setY(3.7);
    const guyGround = lastBase.clone()
      .add(direction.clone().multiplyScalar(2.1)).setY(0.1);
    const guyVector = guyGround.clone().sub(guyTop);
    const guy = new THREE.Mesh(
      new THREE.CylinderGeometry(0.012, 0.012, guyVector.length(), 5),
      steelDarkMaterial,
    );
    guy.position.copy(guyTop).add(guyVector.clone().multiplyScalar(0.5));
    guy.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), guyVector.clone().normalize(),
    );
    scene.add(guy);
    const anchor = shadowed(new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.07, 0.3, 6), concreteMaterial,
    ));
    anchor.position.copy(guyGround).setY(0.08);
    scene.add(anchor);

    // Per-phase attachment points: takeoff strings, then each pole. The
    // center phase rides the pole-top pin, outers ride the crossarm.
    const anchorRows = [];
    anchorRows.push(LATERALS.map((lateral) =>
      direction.clone().multiplyScalar(4.6)
        .add(perpendicular.clone().multiplyScalar(lateral))
        .setY(TAKEOFF_ATTACH_Y)));
    for (const base of poleBases) {
      anchorRows.push(LATERALS.map((lateral) =>
        base.clone()
          .add(perpendicular.clone().multiplyScalar(lateral))
          .setY(lateral === 0 ? CENTER_ATTACH_Y : OUTER_ATTACH_Y)));
    }
    for (let span = 0; span < anchorRows.length - 1; span += 1) {
      for (let phase = 0; phase < LATERALS.length; phase += 1) {
        const { mesh, curve } = conductorSpan(
          anchorRows[span][phase], anchorRows[span + 1][phase],
        );
        mesh.userData.sectionIndex = section;
        mesh.userData.type = "conductor";
        interactiveObjects.push(mesh);
        unit.conductors.push(mesh);
        scene.add(mesh);
        if (LATERALS[phase] === 0) unit.curves.push(curve);
      }
    }

    for (let pulseIndex = 0; pulseIndex < 2; pulseIndex += 1) {
      const pulse = new THREE.Mesh(
        pulseGeometry,
        new THREE.MeshBasicMaterial({
          color: OVERLAY.cyan, transparent: true, opacity: 0.9,
        }),
      );
      scene.add(pulse);
      unit.pulses.push({ mesh: pulse, offset: pulseIndex * 0.5 });
    }
    const beacon = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 12, 12),
      new THREE.MeshBasicMaterial({ color: OVERLAY.cyan }),
    );
    beacon.position.copy(lastBase).setY(CENTER_ATTACH_Y + 0.7);
    beacon.userData.sectionIndex = section;
    beacon.userData.type = "beacon";
    interactiveObjects.push(beacon);
    scene.add(beacon);
    unit.beacon = beacon;

    const label = makeLabelSprite(
      `SEC-${String(section).padStart(2, "0")}`, "loading…", "#dce8ea", true,
    );
    label.position.copy(poleBases[1]).setY(CENTER_ATTACH_Y + 2.2);
    scene.add(label);
    unit.label = label;
    sectionUnits.push(unit);
  }

  // Edge-AI control house with the autoencoder layer stack as a translucent
  // holographic overlay above the roof.
  const houseBase = new THREE.Vector3(8.6, 0, -8.6);
  const { house, mastTip } = makeControlHouse(houseBase);
  scene.add(house);

  const hologramSpecs = [
    { size: 1.5, color: OVERLAY.cyan }, { size: 1.2, color: OVERLAY.cyan },
    { size: 0.9, color: OVERLAY.cyan }, { size: 0.6, color: OVERLAY.amber },
    { size: 0.9, color: 0x5a8fe0 }, { size: 1.2, color: 0x5a8fe0 },
    { size: 1.5, color: 0x5a8fe0 },
  ];
  hologramSpecs.forEach((spec, index) => {
    const layer = new THREE.Mesh(
      new THREE.BoxGeometry(spec.size, 0.22, spec.size),
      new THREE.MeshBasicMaterial({
        color: spec.color, transparent: true, opacity: 0.38,
      }),
    );
    layer.position.copy(houseBase).y = 3.2 + index * 0.34;
    hologramLayers.push(layer);
    scene.add(layer);
  });
  decisionBeacon = new THREE.Mesh(
    new THREE.SphereGeometry(0.26, 12, 12),
    new THREE.MeshBasicMaterial({ color: OVERLAY.cyan }),
  );
  decisionBeacon.position.copy(houseBase).y = 5.9;
  scene.add(decisionBeacon);
  towerPointLight = new THREE.PointLight(0x1fb6a8, 0.5, 18, 2);
  towerPointLight.position.copy(houseBase).y = 5.6;
  scene.add(towerPointLight);
  nnLabelSprite = makeLabelSprite(
    "GridWaveformAutoencoder", "loading…", "#ffb84d", true,
  );
  nnLabelSprite.position.copy(houseBase).y = 6.9;
  scene.add(nnLabelSprite);
  // Overhead service feed from the front gantry to the house mast.
  const feed = conductorSpan(
    new THREE.Vector3(3.5, 4.35, -3.4), mastTip, 0.5,
  );
  scene.add(feed.mesh);

  // One-time shadow render for the freshly built static scene (autoUpdate
  // is off; see init()).
  if (renderer) renderer.shadowMap.needsUpdate = true;
}

function computeFaultedRecord(record, live) {
  if (!live || !live.enabled || live.type === "none") return null;
  const modified = {
    ...record,
    actual: record.actual.map((p) => [...p]),
    reconstruction: record.reconstruction.map((p) => [...p]),
    triggers: [...record.triggers],
  };
  const phases = live.phase === "all" ? [0, 1, 2] : [Number(live.phase)];
  const amp = live.severity * 0.4;
  for (const p of phases) {
    const a = modified.actual[p];
    const len = a.length;
    switch (live.type) {
      case "amplitude_sag":
        for (let i = 0; i < len; i++) a[i] *= (1 - amp);
        break;
      case "amplitude_swell":
        for (let i = 0; i < len; i++) a[i] *= (1 + amp);
        break;
      case "phase_offset": {
        const s = Math.floor(amp * len * 0.3);
        const buf = new Array(len);
        for (let i = 0; i < len; i++) buf[i] = a[(i + s) % len];
        for (let i = 0; i < len; i++) a[i] = buf[i];
        break;
      }
      case "harmonic_injection":
        for (let i = 0; i < len; i++) {
          a[i] += amp * Math.sin(2 * Math.PI * 5 * (i / len));
        }
        break;
      case "noise":
        for (let i = 0; i < len; i++) {
          a[i] += amp * (Math.random() * 2 - 1);
        }
        break;
      case "dc_offset":
        for (let i = 0; i < len; i++) a[i] += amp * 0.5;
        break;
      case "frequency_drift": {
        const drift = amp * 0.08;
        for (let i = 0; i < len; i++) {
          a[i] *= Math.cos(
            2 * Math.PI * (1 + drift * (i / len)) * (len / 512),
          );
        }
        break;
      }
    }
  }
  let totalErr = 0;
  for (let p = 0; p < 3; p++) {
    let pe = 0;
    for (let i = 0; i < modified.actual[p].length; i++) {
      const d = modified.actual[p][i] - modified.reconstruction[p][i];
      pe += d * d;
    }
    totalErr += pe / modified.actual[p].length;
  }
  modified.reconstruction_error = Math.sqrt(totalErr / 3);
  let maxThd = 0;
  for (let p = 0; p < 3; p++) {
    const sig = modified.actual[p];
    const n = sig.length;
    if (n < 16) continue;
    let fRe = 0;
    let fIm = 0;
    for (let i = 0; i < n; i++) {
      const ang = (2 * Math.PI * i) / n;
      fRe += sig[i] * Math.cos(ang);
      fIm += sig[i] * Math.sin(ang);
    }
    fRe /= n;
    fIm /= n;
    const fAmp = Math.hypot(fRe, fIm);
    if (fAmp < 1e-12) {
      maxThd = Math.max(maxThd, 1);
      continue;
    }
    let hPow = 0;
    for (let h = 2; h <= 50; h++) {
      let hRe = 0;
      let hIm = 0;
      for (let i = 0; i < n; i++) {
        const ang = (2 * Math.PI * h * i) / n;
        hRe += sig[i] * Math.cos(ang);
        hIm += sig[i] * Math.sin(ang);
      }
      hRe /= n;
      hIm /= n;
      hPow += hRe * hRe + hIm * hIm;
    }
    maxThd = Math.max(maxThd, Math.sqrt(hPow) / fAmp);
  }
  modified.thd_ratio = maxThd;
  modified.prediction =
    modified.reconstruction_error > thresholdValue ||
    maxThd > thdLimitValue
      ? "anomaly"
      : "normal";
  return modified;
}

function currentEvidence(faultOverride) {
  const appState = window.GridReplay?.state;
  if (!appState?.records.length || !appState.data) {
    return null;
  }
  const record = appState.records[appState.current];
  const useRecord = faultOverride || record;
  const points = useRecord.actual[0].length;
  const visible = Math.max(
    2, Math.min(points, Math.ceil(appState.scan * points)),
  );
  let accumulated = 0;
  for (let phase = 0; phase < 3; phase += 1) {
    for (let index = 0; index < visible; index += 1) {
      const difference =
        useRecord.actual[phase][index] -
        useRecord.reconstruction[phase][index];
      accumulated += difference * difference;
    }
  }
  return {
    record: useRecord,
    section: sectionOf(record.source_index),
    reconPercent: (accumulated / (3 * visible) / thresholdValue) * 100,
    thdPercent: Number.isFinite(useRecord.thd_ratio)
      ? (useRecord.thd_ratio / thdLimitValue) * 100
      : null,
  };
}

function update(data) {
  thresholdValue = Math.max(data.training.calibration.threshold, 1e-12);
  thdLimitValue = Math.max(data.config.thd_limit, 1e-12);
  statsBySection = aggregateSections(data.evaluation.observations, SECTIONS);
  // Screen-reader parity: a static text summary of every section's alert state
  // so the animated scene conveys the same information non-visually.
  const summaryEl = document.getElementById("topology-sections-summary");
  if (summaryEl) {
    const parts = statsBySection.map(
      (stats, section) =>
        `Section ${String(section).padStart(2, "0")}: ${stats.alerts} of ` +
        `${stats.records} records flagged, peak THD ${(stats.maxThd * 100).toFixed(1)}%.`,
    );
    summaryEl.textContent =
      "Feeder section alert summary. " + parts.join(" ");
  }
  if (!scene) {
    if (summaryEl) summaryEl.className = "topology-fallback";
    return;
  }
  sectionUnits.forEach((unit, section) => {
    const stats = statsBySection[section];
    const alerted = stats.alerts > 0;
    unit.beacon.material.color.setHex(alerted ? OVERLAY.red : OVERLAY.cyan);
    retextureSprite(
      unit.label,
      `SEC-${String(section).padStart(2, "0")}`,
      `${stats.alerts}/${stats.records} alerts · THD max ` +
        `${(stats.maxThd * 100).toFixed(1)}%`,
      alerted ? "#ff9d8a" : "#dce8ea",
    );
  });
  retextureSprite(
    nnLabelSprite,
    "GridWaveformAutoencoder",
    `${data.model.parameters.toLocaleString()} params · edge AI unit`,
    "#ffd08a",
  );
  requestRender();
}

function updateHud(evidence) {
  const hud = document.getElementById("topology-hud");
  if (!hud || !evidence) return;
  const appState = window.GridReplay?.state;
  if (!appState?.data) return;
  const thdText = evidence.thdPercent === null
    ? "invalid"
    : `${evidence.thdPercent.toFixed(0)}% of limit`;
  const windowMs = appState.data.replay.time_ms.at(-1) || 0;
  const alarm = evidence.record.prediction === "anomaly" ? "ALARM" : "normal";
  hud.textContent =
    `Replaying test ${evidence.record.sample_index} on SEC-` +
    `${String(evidence.section).padStart(2, "0")} · ` +
    `reconstruction evidence ${evidence.reconPercent.toFixed(0)}% of ` +
    `threshold at t=${(appState.scan * windowMs).toFixed(1)} ms · ` +
    `THD ${thdText} · ${alarm}`;
  // Announce only when the section or verdict changes, so the polite live
  // region is not flooded every animation frame.
  const key = `${evidence.section}:${alarm}`;
  if (key !== lastA11yKey) {
    lastA11yKey = key;
    const live = document.getElementById("topology-a11y");
    if (live) {
      live.textContent =
        `Section ${String(evidence.section).padStart(2, "0")} ${alarm}, ` +
        `test record ${evidence.record.sample_index}.`;
    }
  }
}

function frame() {
  if (!renderer || !scene) return;
  if (!isVisible || document.hidden) return;

  const live = window.GridReplay?.liveState;
  const animating =
    !reducedMotion.matches && Boolean(window.GridReplay?.state.playing);
  const tick = window.GridReplay?.tick ?? 0;
  const tickDelta = Math.max(0, tick - lastTick);
  lastTick = tick;
  if (!animating && !needsRender) return;

  if (animating && orbit.auto && !orbit.dragging) orbit.azimuth += 0.0016;
  camera.position.set(
    orbit.target.x + orbit.radius * Math.sin(orbit.polar) * Math.cos(orbit.azimuth),
    orbit.target.y + orbit.radius * Math.cos(orbit.polar),
    orbit.target.z + orbit.radius * Math.sin(orbit.polar) * Math.sin(orbit.azimuth),
  );
  camera.lookAt(orbit.target);

  // --- Compute evidence, with live fault override when active ---
  liveFaultActive = live?.enabled && live?.type !== "none";
  let faultedRecord = null;
  if (liveFaultActive) {
    const appState = window.GridReplay?.state;
    if (appState?.records.length) {
      faultedRecord = computeFaultedRecord(
        appState.records[appState.current], live,
      );
    }
  }
  const evidence = faultedRecord
    ? currentEvidence(faultedRecord)
    : currentEvidence();

  // --- MSE / THD ratios for tower light ---
  if (evidence) {
    currentMseRatio = Math.min(evidence.reconPercent / 100, 1);
    currentThdRatio = evidence.thdPercent !== null
      ? Math.min(evidence.thdPercent / 100, 1)
      : 0;
  }
  faultSection = evidence ? evidence.section : -1;

  const pulseBlink = animating
    ? 0.72 + 0.28 * Math.sin(performance.now() * 0.006)
    : 1;

  // --- Update feeder sections ---
  sectionUnits.forEach((unit, section) => {
    const active = evidence && evidence.section === section;
    const alerted = statsBySection[section] && statsBySection[section].alerts > 0;
    const isFaulted = liveFaultActive && active;

    // Beacon: scale up when faulted or alerted, color by state
    unit.beacon.scale.setScalar(
      isFaulted ? pulseBlink * 1.5
        : alerted ? pulseBlink * 1.25
          : 1,
    );
    unit.beacon.material.color.setHex(
      isFaulted ? OVERLAY.red : alerted ? OVERLAY.red : OVERLAY.cyan,
    );

    // Conductor material swap for live faults
    for (const mesh of unit.conductors) {
      if (isFaulted) {
        if (mesh.material !== faultConductorMaterial) {
          mesh.material = faultConductorMaterial;
        }
      } else {
        if (mesh.material !== conductorMaterial) {
          mesh.material = conductorMaterial;
        }
      }
    }

    // Energy pulses
    for (const pulse of unit.pulses) {
      if (animating) {
        pulse.offset =
          (pulse.offset + tickDelta * (active ? 0.006 : 0.003)) % 1;
      }
      const spanPosition = pulse.offset * unit.curves.length;
      const spanIndex = Math.min(
        unit.curves.length - 1,
        Math.floor(spanPosition),
      );
      unit.curves[spanIndex].getPointAt(
        Math.min(spanPosition - spanIndex, 1),
        pulse.mesh.position,
      );
      pulse.mesh.material.color.setHex(
        isFaulted ? OVERLAY.red
          : active ? OVERLAY.amber
            : OVERLAY.cyan,
      );
    }

    // Label: show fault detail when live fault active on this section
    if (isFaulted) {
      retextureSprite(
        unit.label,
        `SEC-${String(section).padStart(2, "0")}`,
        `FAULT: ${live.type.replace(/_/g, " ")} sev ${live.severity.toFixed(2)}`,
        "#ff4433",
      );
    } else {
      const stats = statsBySection[section];
      if (stats) {
        retextureSprite(
          unit.label,
          `SEC-${String(section).padStart(2, "0")}`,
          `${stats.alerts}/${stats.records} alerts · THD max ` +
            `${(stats.maxThd * 100).toFixed(1)}%`,
          alerted ? "#ff9d8a" : "#dce8ea",
        );
      }
    }
  });

  // --- Edge-AI tower: decision beacon ---
  if (decisionBeacon && evidence) {
    const isAnomaly = evidence.record.prediction === "anomaly";
    decisionBeacon.material.color.setHex(
      isAnomaly ? OVERLAY.red : OVERLAY.cyan,
    );
    decisionBeacon.scale.setScalar(isAnomaly ? pulseBlink * 1.35 : 1);
  }

  // --- Tower point light: MSE-bound glow ---
  if (towerPointLight) {
    const t = currentMseRatio;
    // Lerp from cyan (0x1f,0xb6,0xa8) to red (0xe6,0x48,0x2e)
    const lr = 0x1f + Math.round((0xe6 - 0x1f) * t);
    const lg = 0xb6 + Math.round((0x48 - 0xb6) * t);
    const lb = 0xa8 + Math.round((0x2e - 0xa8) * t);
    towerPointLight.color.setRGB(lr / 255, lg / 255, lb / 255);
    // Intensity: calm 0.3 to alarming 3.0
    towerPointLight.intensity = 0.3 + t * 2.7;
    if (animating) {
      towerPointLight.intensity *=
        0.85 + 0.15 * Math.sin(performance.now() * 0.004);
    }
  }

  // --- Hologram layers: color shift with MSE ---
  for (const layer of hologramLayers) {
    const t = currentMseRatio;
    layer.material.color.setHex(t < 0.5 ? OVERLAY.cyan : OVERLAY.amber);
    layer.material.opacity = 0.25 + t * 0.4;
  }

  updateHud(evidence);
  renderer.render(scene, camera);
  needsRender = false;
}

function resize() {
  if (!renderer || !camera || !canvasElement) return;
  const width = canvasElement.clientWidth || canvasElement.width;
  const height = canvasElement.clientHeight || canvasElement.height;
  if (!width || !height) return;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  // Narrow / portrait viewports: pull the camera back and grow labels so the
  // whole substation frames and the text stays legible on a phone.
  const aspect = width / height;
  orbit.radius = aspect < 0.8 ? 46 : aspect < 1.2 ? 40 : 34;
  const labelScale = aspect < 0.8 ? 1.7 : aspect < 1.2 ? 1.3 : 1;
  for (const sprite of labelSprites) {
    const [baseW, baseH] = sprite.userData.baseScale;
    sprite.scale.set(baseW * labelScale, baseH * labelScale, 1);
  }
  requestRender();
}

function attachControls() {
  canvasElement.addEventListener("pointerdown", (event) => {
    orbit.dragging = true;
    orbit.lastX = event.clientX;
    orbit.lastY = event.clientY;
    try {
      canvasElement.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic pointer events have no active pointer to capture.
    }
  });
  canvasElement.addEventListener("pointermove", (event) => {
    if (!orbit.dragging) return;
    orbit.azimuth += (event.clientX - orbit.lastX) * 0.006;
    orbit.polar = Math.max(
      0.3,
      Math.min(1.5, orbit.polar - (event.clientY - orbit.lastY) * 0.005),
    );
    orbit.lastX = event.clientX;
    orbit.lastY = event.clientY;
    requestRender();
  });
  const release = () => {
    orbit.dragging = false;
  };
  canvasElement.addEventListener("pointerup", release);
  canvasElement.addEventListener("pointercancel", release);
  canvasElement.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      orbit.radius = Math.max(
        14, Math.min(60, orbit.radius + event.deltaY * 0.03),
      );
      requestRender();
    },
    { passive: false },
  );
  canvasElement.addEventListener("keydown", (event) => {
    const handled = [
      "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-",
    ].includes(event.key);
    if (!handled) return;
    event.preventDefault();
    if (event.key === "ArrowLeft") orbit.azimuth -= 0.08;
    if (event.key === "ArrowRight") orbit.azimuth += 0.08;
    if (event.key === "ArrowUp") orbit.polar = Math.max(0.3, orbit.polar - 0.06);
    if (event.key === "ArrowDown") orbit.polar = Math.min(1.5, orbit.polar + 0.06);
    if (event.key === "+" || event.key === "=") {
      orbit.radius = Math.max(14, orbit.radius - 2);
    }
    if (event.key === "-") orbit.radius = Math.min(60, orbit.radius + 2);
    requestRender();
  });
}

// ---------------------------------------------------------------------------
// Raycasting: click a feeder section or beacon to select it
// ---------------------------------------------------------------------------

function onCanvasClick(event) {
  if (!renderer || !scene || !camera) return;
  const rect = canvasElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(interactiveObjects, false);
  if (hits.length === 0) return;
  const hit = hits[0].object;
  const section = hit.userData.sectionIndex;
  if (section === undefined || section < 0) return;
  document.dispatchEvent(
    new CustomEvent("grid-sentinel:section-click", {
      detail: { section },
    }),
  );
}

function showFallback(message) {
  const hud = document.getElementById("topology-hud");
  if (hud) hud.textContent = message;
  if (canvasElement) canvasElement.style.display = "none";
  const summaryEl = document.getElementById("topology-sections-summary");
  if (summaryEl && statsBySection.length) {
    // Promote the offscreen summary to a visible fallback list.
    summaryEl.className = "topology-fallback";
  }
}

function webglAvailable() {
  try {
    const probe = document.createElement("canvas");
    return !!(
      window.WebGLRenderingContext &&
      (probe.getContext("webgl") || probe.getContext("experimental-webgl"))
    );
  } catch (error) {
    return false;
  }
}

function init() {
  canvasElement = document.getElementById("topology-canvas");
  if (!canvasElement) return;
  if (!webglAvailable()) {
    showFallback(
      "3D view unavailable (no WebGL). Section alert data is available to " +
        "assistive technology and in the panels below.",
    );
    return;
  }
  try {
    renderer = new THREE.WebGLRenderer({ canvas: canvasElement, antialias: true });
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    // Every shadow caster is static; render the shadow map once per scene
    // build instead of on every animated frame.
    renderer.shadowMap.autoUpdate = false;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    camera = new THREE.PerspectiveCamera(45, 2, 0.1, 300);
    buildStaticScene();
    attachControls();
    canvasElement.addEventListener("click", onCanvasClick);
    resize();

    // Pause rendering when the panel scrolls out of view, resume (and draw one
    // frame) when it returns.
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          isVisible = entries[0].isIntersecting;
          if (isVisible) requestRender();
        },
        { threshold: 0.01 },
      );
      observer.observe(canvasElement);
    }
    if ("ResizeObserver" in window) {
      let scheduled = false;
      const ro = new ResizeObserver(() => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
          scheduled = false;
          resize();
        });
      });
      ro.observe(canvasElement);
    } else {
      window.addEventListener("resize", resize);
    }
    reducedMotion.addEventListener?.("change", requestRender);
    canvasElement.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
    });
    canvasElement.addEventListener("webglcontextrestored", () => {
      buildStaticScene();
      if (window.GridReplay?.state.data) update(window.GridReplay.state.data);
      resize();
      requestRender();
    });

    if (window.GridReplay?.state.data) update(window.GridReplay.state.data);
    requestRender();
    frame();
  } catch (error) {
    showFallback(
      "3D view could not start. Section alert data is available to assistive " +
        "technology and in the panels below.",
    );
  }
}

window.GridTopology = { update, frame };
init();
