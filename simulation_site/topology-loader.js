// Lazy-loads the 3D scene module (Three.js is ~0.7 MB) only when its panel
// nears the viewport, so first paint is not blocked by it. Kept as a separate
// same-origin file because the dashboard CSP forbids inline scripts.
const canvas = document.getElementById("topology-canvas");
let loaded = false;

function load() {
  if (loaded) return;
  loaded = true;
  import("./topology3d.js");
}

if (canvas && "IntersectionObserver" in window) {
  const io = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        io.disconnect();
        load();
      }
    },
    { rootMargin: "400px" },
  );
  io.observe(canvas);
} else {
  load();
}
