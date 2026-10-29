#import "@preview/cetz:0.4.2"
#import "style.typ": blue, body-font, heading-font, ink, muted, paper, rule
#import "_layers.typ": layer-color, layer-frame, layer-label

#set page(width: 190mm, height: auto, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

// Cross-sections of the QPDK layer stacks, not to scale. Every number, colour
// and GDS layer comes from data: the levels from layer-stack.yaml (checked
// against qpdk/tech.py by tests/test_layer_stack_figure.py) and the colours
// from qpdk/layers.yaml. To add a level, add it to tech.py and the YAML, then
// draw it below with `slab` (or `slotted` for an etched metal) and give it a
// `callout`. The legend and its numbering follow on their own.

// ── Data ─────────────────────────────────────────────────────────────────────

#let stacks = yaml("layer-stack.yaml").stacks

// Legend columns, each a list of stacks from top to bottom. Legend numbers run
// through the columns in this order.
#let legend-columns = (
  ("LAYER_STACK", "LAYER_STACK_FLIP_CHIP"),
  (
    "LAYER_STACK_BACKSIDE",
    "LAYER_STACK_FLIP_CHIP_BACKSIDE",
    "LAYER_STACK_GATEMON",
  ),
)
#for name in stacks.keys() {
  assert(
    name in legend-columns.flatten(),
    message: name + " has no legend column",
  )
}

// Legend number of each listed level, by level name.
#let numbers = {
  let out = (:)
  for name in legend-columns.flatten() {
    for level in stacks.at(name).levels.keys() {
      out.insert(level, out.len() + 1)
    }
  }
  out
}

// Level `name` as `stack` defines it: its own levels and reuses first, then
// the stacks it extends. Returns the YAML entry plus its name.
#let find-level(stack, name) = {
  let entry = stacks.at(stack)
  for group in ("levels", "reuses") {
    let levels = entry.at(group, default: (:))
    if name in levels { return levels.at(name) + (name: name) }
  }
  for parent in entry.at("extends", default: ()) {
    let found = find-level(parent, name)
    if found != none { return found }
  }
  none
}
#let level(stack, name) = {
  let found = find-level(stack, name)
  assert(found != none, message: name + " is not a level of " + stack)
  found
}

// ── Styling and text ─────────────────────────────────────────────────────────

#let dashed = (paint: muted, thickness: 0.6pt, dash: "dashed")
// Neutral looks for levels drawn on SIM_AREA, instead of its hidden green.
#let looks = (
  substrate: (fill: paper, stroke: 0.7pt + rule),
  vacuum: (fill: white, stroke: dashed),
  oxide: (fill: rule, stroke: none),
)
#let look(lv) = {
  let color = lv.at("color", default: lv.layer)
  if color in looks { return looks.at(color) }
  let shade = lv.at("shade", default: 0) * 1%
  (
    fill: layer-color(color).darken(shade),
    stroke: 0.5pt + layer-frame(color).darken(shade),
  )
}

#let small(body) = text(size: 6.5pt, fill: muted, body)
#let minus(s) = s.replace("-", "−")
// 0.2 → "200 nm", -500.2 → "−500.2 µm"
#let length(um) = if um != 0 and calc.abs(um) < 1 {
  minus(str(calc.round(um * 1000))) + " nm"
} else { minus(str(calc.round(um, digits: 1))) + " µm" }
#let formulas = (SiO2: [SiO#sub[2]])
#let material(name) = formulas.at(name, default: name)
#let spec(lv) = {
  let note = lv.at("note", default: none)
  let thickness = length(lv.thickness) + if note != none { " (" + note + ")" }
  let z = if lv.zmin == 0 { "0" } else { length(lv.zmin) }
  let source = if lv.at("derived", default: false) {
    "derived from " + lv.layer
  } else { layer-label(lv.layer) }
  [#material(lv.material) · #thickness · $z$ #z · #source]
}

// ── Broken z axis ────────────────────────────────────────────────────────────

// Maps z (µm) to canvas y (cm) piecewise, so nanometre films and 500 µm
// substrates share one panel. Segments are (z-from, z-to, height), plus
// `true` for a shortened segment, which slabs crossing it mark with a break.
// The point `z` lands on canvas height `y`.
#let broken-axis(segments, z: 0, y: 0) = {
  let starts = segments
    .map(s => s.at(2))
    .fold((0,), (acc, h) => acc + (acc.last() + h,))
  let raw(v) = {
    for (i, s) in segments.enumerate() {
      let (a, b, h) = s.slice(0, 3)
      if v <= b or i == segments.len() - 1 {
        return starts.at(i) + (calc.clamp(v, a, b) - a) / (b - a) * h
      }
    }
  }
  let offset = y - raw(z)
  (
    y: v => offset + raw(v),
    breaks: segments
      .enumerate()
      .filter(((i, s)) => s.len() > 3 and s.at(3))
      .map(((i, s)) => offset + starts.at(i) + s.at(2) / 2),
  )
}

// ── Drawing helpers ──────────────────────────────────────────────────────────

#let min-film = 0.06 // thinnest drawn film, cm, so 15 nm stays visible

// Canvas (bottom, top) of a level on an axis.
#let span(ax, lv) = {
  let lo = (ax.y)(lv.zmin)
  (lo, calc.max((ax.y)(lv.zmin + lv.thickness), lo + min-film))
}

#let break-band(y, from, to) = {
  import cetz.draw: *
  rect((from, y - 0.13), (to, y + 0.13), fill: white, stroke: none)
  let steps = calc.max(1, int((to - from) / 0.3))
  for dy in (-0.13, 0.13) {
    line(
      ..range(steps + 1).map(i => (
        from + i * (to - from) / steps,
        y + dy + if calc.even(i) { 0.05 } else { -0.05 },
      )),
      stroke: 0.6pt + muted,
    )
  }
}

// A level drawn between x = from and to, broken where the axis is shortened.
#let slab(ax, lv, from, to) = {
  import cetz.draw: *
  let (lo, hi) = span(ax, lv)
  rect((from, lo), (to, hi), ..look(lv))
  for y in ax.breaks.filter(y => lo < y and y < hi) { break-band(y, from, to) }
}

// An etched metal: the level between from and to, cut by `gaps` (x pairs).
#let slotted(ax, lv, from, to, gaps) = {
  let edges = (from, ..gaps.flatten(), to)
  for i in range(0, edges.len(), step: 2) {
    slab(ax, lv, edges.at(i), edges.at(i + 1))
  }
}

// Width arrows across each gap at height y, with a label centred under them.
#let gap-label(gaps, y, label) = {
  import cetz.draw: *
  for (a, b) in gaps {
    line((a, y), (b, y), stroke: 0.6pt + muted, mark: (
      start: ">",
      end: ">",
      size: 0.08,
      fill: muted,
    ))
  }
  let mid = (gaps.first().first() + gaps.last().last()) / 2
  content((mid, y - 0.27), small(label), anchor: "center")
}

#let badge(at, n) = {
  import cetz.draw: *
  circle(at, radius: 0.17, fill: white, stroke: 0.6pt + ink)
  content(at, text(size: 6.5pt, weight: "bold", str(n)), anchor: "center")
}

// Ties a drawn level to its legend number: a leader from (x, y) on the level,
// by default its middle, to the numbered badge at `to`, by default in place.
#let callout(ax, lv, x, y: auto, to: auto) = {
  import cetz.draw: *
  let (lo, hi) = span(ax, lv)
  let at = (x, if y == auto { (lo + hi) / 2 } else { y })
  let label-at = if to == auto { at } else { to }
  line(at, label-at, stroke: 0.5pt + muted)
  badge(label-at, numbers.at(lv.name))
}

// A z marker on the panel edge at x, labelled with z itself.
#let tick(ax, z, x, side: "left") = {
  import cetz.draw: *
  let y = (ax.y)(z)
  let label = small(if z == 0 [$z = 0$] else { length(z) })
  let (dx, anchor) = if side == "left" { (-0.15, "east") } else {
    (0.15, "west")
  }
  line((x, y), (x + dx, y), stroke: 0.6pt + muted)
  content((x + 1.3 * dx, y), label, anchor: anchor)
}

#let panel-heading(at, body) = cetz.draw.content(
  at,
  text(size: 7pt, font: heading-font, weight: "bold", body),
  anchor: "center",
)

// ── Figure ───────────────────────────────────────────────────────────────────

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  // Main panel: both chips of LAYER_STACK_FLIP_CHIP_BACKSIDE, which holds
  // every level of the base, flip-chip and backside stacks. Films at
  // 1 cm = 400 nm; substrates, vacuum and the indium bump are shortened.
  let main = "LAYER_STACK_FLIP_CHIP_BACKSIDE"
  let lv(name) = level(main, name)
  let ax = broken-axis((
    (-1000.2, -500.2, 1.6, true), // Vacuum_bottom
    (-500.2, -500, 0.5), // MB
    (-500, 0, 3.0, true), // Substrate
    (0, 0.5, 1.25), // bottom-chip films
    (0.5, 10, 2.4, true), // indium bump gap
    (10, 10.2, 0.5), // M2
    (10.2, 510.2, 1.8, true), // Substrate_top
    (510.2, 510.4, 0.5), // MB2
    (510.4, 1010.4, 1.6, true), // Vacuum_top
  ))
  let x0 = 1.3
  let x1 = 11.8
  let chip-x = 6.6 // left edge of the top chip
  let row1 = (ax.y)(0.5) + 0.35 // callout rows above the bottom-chip films
  let row2 = row1 + 0.45

  // Bottom chip, from the vacuum down. The vacuum spans the gap under the top
  // chip too; the top chip is drawn over it.
  slab(ax, lv("Vacuum"), x0, x1)
  slab(ax, lv("Vacuum_bottom"), x0, x1)
  slotted(ax, lv("MB"), x0, x1, ((3.5, 4.0), (5.0, 5.5)))
  gap-label(
    ((3.5, 4.0), (5.0, 5.5)),
    (ax.y)(-500.2) - 0.2,
    [MB_ETCH (3/1) slots],
  )
  slab(ax, lv("Substrate"), x0, x1)
  content(
    (6.4, (ax.y)(-150)),
    text(size: 10pt, fill: muted, [Substrate: Si, $ε_upright(r) = 11.45$]),
    anchor: "center",
  )
  slab(ax, lv("TSV"), 1.85, 2.35)

  // CPW on M1: ground, slot, signal, slot, ground.
  slotted(ax, lv("M1"), x0, 7.2, ((3.5, 4.0), (5.0, 5.5)))
  gap-label(((3.5, 4.0), (5.0, 5.5)), (ax.y)(0) - 0.18, [M1_ETCH (1/1) slots])
  // Airbridge over the signal line, landing on both grounds through vias.
  for (a, b) in ((2.75, 3.15), (5.85, 6.25)) {
    slab(ax, lv("Airbridge_Via"), a, b)
  }
  slab(ax, lv("Airbridge"), 2.75, 6.25)
  // Josephson junction bridging a gap between two M1 electrodes.
  slab(ax, lv("M1"), 7.65, 8.35)
  slab(ax, lv("JosephsonJunction"), 7.05, 7.8)
  slab(ax, lv("NbTiN"), 8.8, 9.7)
  // Indium bump from an M1 pad to the M2 pad above.
  slab(ax, lv("M1"), 10.1, 11.5)
  slab(ax, lv("IndiumBump"), 10.35, 11.25)

  // Top chip: M2 faces down at 10 µm, TSV2 runs up through Substrate_top to
  // MB2 on its outer face, under Vacuum_top.
  slotted(ax, lv("M2"), chip-x, x1, ((9.75, 10.1),))
  content((9.88, (ax.y)(10) - 0.2), small[M2_ETCH (2/1)], anchor: "north-east")
  slab(ax, lv("Substrate_top"), chip-x, x1)
  slab(ax, lv("TSV2"), 7.3, 7.8)
  slotted(ax, lv("MB2"), chip-x, x1, ((8.3, 8.8), (9.8, 10.3)))
  slab(ax, lv("Vacuum_top"), chip-x, x1)
  gap-label(
    ((8.3, 8.8), (9.8, 10.3)),
    (ax.y)(510.4) + 0.5,
    [MB2_ETCH (4/1) slots],
  )

  for z in (0, 0.2, -500, -500.2) { tick(ax, z, x0) }
  for z in (10, 10.2, 510.2, 510.4) { tick(ax, z, x1, side: "right") }

  let vacuum-y = (ax.y)(350)
  content(
    (x0 + 0.25, vacuum-y),
    text(size: 8pt, fill: muted, [Vacuum]),
    anchor: "west",
  )
  callout(ax, lv("Vacuum"), 3.0, y: vacuum-y)
  callout(ax, lv("M1_ground"), 1.9, to: (1.9, row1))
  callout(ax, lv("M1"), 4.5, to: (4.5, row1))
  callout(ax, lv("Airbridge_Via"), 6.05, to: (6.6, row1))
  callout(ax, lv("Airbridge"), 3.6, to: (3.6, row2))
  callout(ax, lv("JosephsonJunction"), 7.3, to: (7.3, row1))
  callout(ax, lv("NbTiN"), 9.25, to: (9.25, row1))
  callout(ax, lv("IndiumBump"), 10.35, y: row1, to: (9.85, row1))
  let substrate-y = (ax.y)(-420)
  callout(ax, lv("TSV"), 2.1, y: substrate-y, to: (2.75, substrate-y))
  callout(ax, lv("Substrate"), 9.6, y: substrate-y)
  let mb-label = (ax.y)(-500.2) - 0.4
  callout(ax, lv("MB"), 7.5, to: (7.5, mb-label))
  callout(ax, lv("Vacuum_bottom"), 9.6, y: (ax.y)(-900))
  let top-row = (ax.y)(110)
  callout(ax, lv("M2_ground"), 8.4, to: (8.4, top-row))
  callout(ax, lv("M2"), 10.9, to: (10.9, top-row))
  let top-substrate-y = (ax.y)(430)
  callout(ax, lv("TSV2"), 7.55, y: top-substrate-y, to: (8.2, top-substrate-y))
  callout(ax, lv("Substrate_top"), 9.6, y: top-substrate-y)
  callout(ax, lv("MB2"), 7.0, to: (7.0, (ax.y)(510.4) + 0.45))
  callout(ax, lv("Vacuum_top"), 11.1, y: (ax.y)(900))

  // Gatemon panel: z = 0 is the oxide surface, films at 1 cm = 50 nm.
  let gm = "LAYER_STACK_GATEMON"
  let lv(name) = level(gm, name)
  let ag = broken-axis(
    (
      (-500.15, -0.15, 0.6, true), // Substrate
      (-0.15, 0.2, 7.0), // oxide and films
      (0.2, 500, 0.7, true), // Vacuum
    ),
    y: (ax.y)(0),
  )
  let gx0 = 13.3
  let gx1 = 18.4
  let gx = (gx0 + gx1) / 2
  let film-row = (ag.y)(0.19)
  panel-heading((gx, (ag.y)(500) + 0.6), [Gatemon films])
  content(
    (gx, (ag.y)(500) + 0.25),
    small[$z = 0$ at the oxide surface],
    anchor: "center",
  )

  slab(ag, lv("Vacuum"), gx0, gx1)
  slab(ag, lv("Substrate"), gx0, gx1)
  slab(ag, lv("Oxide"), gx0, gx1)
  // Nanotube between the contacts: not extruded, shown for orientation.
  let surface = (ag.y)(0)
  line((14.6, surface + 0.03), (17.1, surface + 0.03), stroke: (
    paint: layer-color("CNT"),
    thickness: 0.8pt,
    dash: "dotted",
  ))
  content(
    (15.85, surface - 0.25),
    small[nanotube (not extruded)],
    anchor: "center",
  )
  for (a, b) in ((14.2, 15.0), (16.7, 17.5)) {
    slab(ag, lv("ContactNb"), a, b)
    slab(ag, lv("ContactAu"), a, b)
  }
  for (a, b) in ((13.6, 14.6), (17.1, 18.1)) {
    slab(ag, lv("Reconnection"), a, b)
  }
  for (a, b) in ((gx0, 13.85), (17.85, gx1)) { slab(ag, lv("M1"), a, b) }
  slab(ag, lv("hBN"), 14.85, 16.85)
  slab(ag, lv("TopGate"), 15.35, 16.35)

  let oxide-row = (ag.y)(-0.065)
  callout(ag, lv("M1"), 13.6, to: (13.6, film-row))
  callout(ag, lv("Oxide"), 15.85, y: (ag.y)(-0.11))
  callout(ag, lv("ContactNb"), 14.35, to: (14.35, oxide-row))
  callout(ag, lv("ContactAu"), 17.35, to: (17.35, oxide-row))
  callout(ag, lv("hBN"), 16.6, to: (16.95, film-row))
  callout(ag, lv("TopGate"), 15.85)
  callout(ag, lv("Reconnection"), 17.6, to: (17.6, film-row))

  // Title, above the taller panel.
  let top = calc.max((ax.y)(1010.4), (ag.y)(500) + 0.6) + 0.7
  content(
    (0.5, top + 0.45),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [QPDK LAYER STACKS],
    ),
    anchor: "west",
  )
  content(
    (0.5, top),
    text(
      size: 8pt,
      fill: muted,
      [Cross-sections, not to scale. Every level of every stack is numbered and listed below.],
    ),
    anchor: "west",
  )

  // Legend: one group per stack, flowing down each column.
  let row-h = 0.7
  let legend-row(x, y, lv) = {
    badge((x + 0.17, y), numbers.at(lv.name))
    rect((x + 0.52, y - 0.17), (x + 0.92, y + 0.17), ..look(lv))
    content(
      (x + 1.12, y + 0.14),
      text(size: 7.5pt, font: heading-font, weight: "bold", lv.name),
      anchor: "west",
    )
    content((x + 1.12, y - 0.18), small(spec(lv)), anchor: "west")
  }
  let legend-top = (ax.y)(-1000.2) - 0.6
  for (column, names) in legend-columns.enumerate() {
    let x = 0.5 + column * 9.4
    let y = legend-top
    for name in names {
      let entry = stacks.at(name)
      content(
        (x, y),
        text(
          size: 7pt,
          font: heading-font,
          weight: "bold",
          fill: blue,
          name + " " + entry.title,
        ),
        anchor: "west",
      )
      y -= 0.55
      for level-name in entry.levels.keys() {
        legend-row(x, y, level(name, level-name))
        y -= row-h
      }
      y -= 0.15
    }
  }
})
