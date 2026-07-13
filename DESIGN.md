---
name: Grid Sentinel
description: A measured control-room interface that connects waveform evidence to detector decisions.
colors:
  telemetry-cyan: "oklch(0.82 0.14 180)"
  telemetry-cyan-deep: "oklch(0.62 0.13 183)"
  phase-blue: "oklch(0.76 0.12 250)"
  anomaly-amber: "oklch(0.84 0.15 75)"
  alarm-red: "oklch(0.70 0.19 28)"
  verified-green: "oklch(0.78 0.14 152)"
  focus-yellow: "oklch(0.90 0.16 95)"
  control-room-bg: "oklch(0.13 0.014 190)"
  instrument-surface: "oklch(0.175 0.018 190)"
  raised-surface: "oklch(0.21 0.021 190)"
  signal-line: "oklch(0.34 0.025 190)"
  signal-line-soft: "oklch(0.27 0.021 190)"
  primary-ink: "oklch(0.95 0.012 175)"
  secondary-ink: "oklch(0.75 0.025 180)"
typography:
  headline:
    fontFamily: "Segoe UI Variable, Aptos, system-ui, sans-serif"
    fontSize: "2.65rem"
    fontWeight: 850
    lineHeight: 1.08
    letterSpacing: "-0.025em"
  title:
    fontFamily: "Segoe UI Variable, Aptos, system-ui, sans-serif"
    fontSize: "1.32rem"
    fontWeight: 700
    lineHeight: 1.2
  body:
    fontFamily: "Segoe UI Variable, Aptos, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "Cascadia Code, ui-monospace, monospace"
    fontSize: "0.75rem"
    fontWeight: 700
rounded:
  bar: "2px"
  control: "9px"
  panel: "14px"
spacing:
  xs: "0.35rem"
  sm: "0.65rem"
  md: "1rem"
  lg: "1.5rem"
  xl: "2.25rem"
  section: "3.5rem"
components:
  button-primary:
    backgroundColor: "{colors.telemetry-cyan}"
    textColor: "{colors.control-room-bg}"
    rounded: "{rounded.control}"
    padding: "0 0.95rem"
    height: "44px"
  button-quiet:
    backgroundColor: "{colors.instrument-surface}"
    textColor: "{colors.primary-ink}"
    rounded: "{rounded.control}"
    padding: "0 0.95rem"
    height: "44px"
  instrument-panel:
    backgroundColor: "{colors.instrument-surface}"
    textColor: "{colors.primary-ink}"
    rounded: "{rounded.panel}"
    padding: "1.5rem"
  telemetry-readout:
    backgroundColor: "{colors.raised-surface}"
    textColor: "{colors.primary-ink}"
    rounded: "{rounded.control}"
    padding: "0.7rem 1rem"
---

# Design System: Grid Sentinel

## Overview

**Creative North Star: "The Substation Oscilloscope"**

Grid Sentinel should feel like a trusted instrument in a dim operations room:
dense enough for an engineer, restrained enough to keep the signal primary, and
clear enough for a reviewer to understand why an alert fired. The design uses
measured telemetry as its visual material. Motion communicates replay position and
state changes only.

The system rejects generic SaaS metric-card grids, decorative glass effects, fake
“live” data, unreadable neon-on-black styling, mojibake, and unexplained perfect
scores. Every impressive result must remain adjacent to provenance or a limitation.

The instrument view now has a physical counterpart: **The Field Model**, a
daylight-rendered 3D substation and feeder scene reached from the same page.
It is the walkable equipment the oscilloscope is measuring, not a second
aesthetic. Detection state rides on it as an AR-style overlay (beacons, energy
pulses, evidence readout) so the same evidence color rule still governs it.

**Key Characteristics:**

- Signal-first information hierarchy
- Restrained control-room palette with semantic anomaly colors
- Flat structural panels and crisp measurement lines
- Familiar, keyboard-accessible controls
- Explicit recorded-run provenance
- One realistic 3D field counterpart to the instrument view, not a second style

## Colors

Telemetry cyan carries normal system state and primary action. Amber identifies
power-quality concern, red is reserved for a detector alarm, and phase blue keeps
three-phase plots distinguishable without turning the whole surface colorful.

### Primary

- **Telemetry Cyan:** primary controls, live traces, focus-adjacent state, and
  successful detector flow.
- **Deep Telemetry Cyan:** control borders and quieter active states.

### Secondary

- **Phase Blue:** Phase B traces and comparison series only.
- **Anomaly Amber:** threshold limits, cautions, and explicit synthetic-data notes.
- **Alarm Red:** actual anomaly predictions and error states only.
- **Focus Yellow:** the single, deliberately off-palette color reserved for
  keyboard focus. It never appears as decoration or state; it exists to be
  the one thing your eye can't miss when tabbing through controls.

### Neutral

- **Control-Room Background:** the application canvas.
- **Instrument Surface:** primary panels.
- **Raised Surface:** decision rails, hover rows, and nested operational controls.
- **Signal Line / Signal Line Soft:** one-pixel structural borders; the soft
  variant recedes for secondary divisions inside a panel.
- **Primary Ink / Secondary Ink:** high-contrast data and supporting labels.

**The Evidence Color Rule.** Saturated color always encodes a phase, state, or
decision. Decorative accent color is forbidden.

## Typography

**Display Font:** Segoe UI Variable (with Aptos and system fallbacks)  
**Body Font:** Segoe UI Variable (with Aptos and system fallbacks)  
**Label/Mono Font:** Cascadia Code (with generic monospace fallback)

**Character:** The single humanist sans family keeps the interface familiar and
operational. Monospace is limited to indices, hashes, numerical evidence, and the
small section labels used to orient dense technical content.

### Hierarchy

- **Headline** (850, 2.65rem, 1.08): one page-level statement, never a metric.
- **Title** (700, 1.32rem, 1.2): panel and workflow headings.
- **Body** (400, 1rem, 1.5): explanation with a 65–75ch line-length target.
- **Label** (700, 0.75rem): compact technical orientation and provenance.

**The Instrument Label Rule.** Monospace means machine evidence. It must not be
used as a decorative “technical” texture.

## Elevation

The system is flat by default and uses tonal layers plus one-pixel structural
borders. There are no decorative drop shadows. The waveform stage is darker
because it is an actual measurement canvas, while the decision rail is slightly
lighter to establish reading order.

**The Flat Control Rule.** If a panel needs a broad shadow to appear separate, the
layout or tonal hierarchy is wrong.

## Components

### Buttons

- **Shape:** compact, confident corners (9px) with a 44px minimum target.
- **Primary:** telemetry cyan with control-room text.
- **Hover / Focus:** ink-colored hover and a visible high-contrast focus ring.
- **Quiet:** transparent or tonal surface with a structural border.

### Cards / Containers

- **Corner Style:** restrained instrument-panel curve (14px).
- **Background:** instrument or raised surface according to hierarchy.
- **Shadow Strategy:** none; borders and tonal contrast carry depth.
- **Internal Padding:** 1rem on small screens and 1.5rem at normal density.

### Inputs / Fields

- **Style:** native controls with the same 9px control radius and a 44px target.
- **Focus:** three-pixel yellow focus outline, offset from the control.
- **Disabled:** opacity reduction only after the text state also explains why.

### Navigation

The sticky top bar owns run status and replay controls. It collapses into two rows
at tablet width and becomes a non-sticky single column on small screens.

### Signal Workspace

Actual phases are solid cyan, blue, and amber. Reconstruction is a thin dashed
neutral line. A moving vertical cursor is the only continuous decorative motion,
and reduced-motion mode disables automatic replay.

### Telemetry Readout

A raised-surface strip of monospace label/value pairs (signal time, 60 Hz
cycles, clock mode, tick, dropped ticks) reporting the fixed-step simulation
clock, and a matching one-line status for adaptive polling ("Data current ·
next check in ~Ns" / "Unchanged · next check in ~Ns"). Both are machine
evidence, not decoration, so they follow the Instrument Label Rule: monospace,
compact, never centered for effect.

### The Field Model (Signature Component)

The realistic 3D grid scene: a daylight-rendered substation (fenced gravel
pad, transformers, aluminum busbars on porcelain insulators) and eight wooden
feeder lines with sagging conductors and real cast shadows, orbited by drag
and scroll. It is deliberately photorealistic-leaning where the rest of the
system is flat and instrument-like, because it represents the physical
equipment, not the control room reading it. Detection state overlays it in
the system's own semantic colors: cyan/red status beacons per section, amber
energy pulses that travel the conductors in step with the simulation clock,
and a decision beacon on the edge-AI control house that goes alarm-red on an
active anomaly. The autoencoder's layer stack renders above the control house
as a translucent hologram, not a solid object, so it still reads as an
overlay rather than invented physical hardware. The evidence readout beneath
the scene updates every clock tick with the same replay percentages shown in
the Decision Gate, so the Field Model and the oscilloscope never disagree.

**The One Physical World Rule.** The Field Model shows one realistic scene,
not a set of interchangeable renders. Detection color, alert language, and
recorded-vs-live labeling must match the control-room instrument view
exactly; the Field Model is a second viewport onto the same evidence, never a
separate visual system.

## Do's and Don'ts

### Do:

- **Do** connect every prediction to PLL-normalized waveform reconstruction,
  THD, and threshold evidence; identify the Fortescue penalty as a training
  constraint rather than a third alert gate.
- **Do** use text and line style in addition to color for normal/anomaly state.
- **Do** retain 44px controls, visible focus, semantic tables, and reduced motion.
- **Do** label completed data as “Recorded replay,” never as a live sensor feed.
- **Do** keep the synthetic-data limitation adjacent to perfect metrics.
- **Do** keep the Field Model's detection overlay (beacons, pulses, evidence
  readout) synchronized to the same simulation clock and calibration data as
  every other panel; it is a second viewport, not a second dataset.

### Don't:

- **Don't** use generic SaaS metric-card grids.
- **Don't** use decorative glass effects, broad ghost-card shadows, or purple
  gradients.
- **Don't** show fake “live” data or unexplained perfect scores.
- **Don't** use unreadable neon-on-black styling or color alone for alarms.
- **Don't** allow mojibake, copied metric drift, or stale report values.
- **Don't** use side-stripe borders, gradient text, or oversized rounded panels.
