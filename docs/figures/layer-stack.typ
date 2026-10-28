#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 190mm, height: 96mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let copper = rgb("#a95a20")

// Cross-section of `qpdk.tech.LAYER_STACK`. Not to scale: the thin films use
// 1 cm = 400 nm, while the 500 µm substrate and vacuum and the 10 µm indium bump
// are drawn shortened behind break marks.
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let nm = 1 / 400
  let z0 = 4.6
  let z(v) = z0 + v * nm
  let x0 = 1.3
  let x1 = 11.8

  content(
    (0.5, 9.95),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [QPDK LAYER STACK],
    ),
    anchor: "west",
  )
  content(
    (0.5, 9.5),
    text(
      size: 8pt,
      fill: muted,
      [Cross-section, not to scale. $z = 0$ is the top of the substrate; all films are listed on the right.],
    ),
    anchor: "west",
  )

  let break-band(y, from, to) = {
    rect((from, y - 0.13), (to, y + 0.13), fill: white, stroke: none)
    let steps = int((to - from) / 0.3)
    let zig(dy) = line(
      ..range(steps + 1).map(i => (
        from + i * (to - from) / steps,
        y + dy + (if calc.even(i) { 0.05 } else { -0.05 }),
      )),
      stroke: 0.6pt + muted,
    )
    zig(-0.13)
    zig(0.13)
  }

  // Vacuum above the metal and the silicon substrate below it.
  rect(
    (x0, z(200)),
    (x1, 8.85),
    fill: white,
    stroke: (paint: muted, thickness: 0.6pt, dash: "dashed"),
  )
  break-band(7.85, x0, x1)
  content(
    (x0 + 0.25, 8.45),
    text(size: 8pt, fill: muted, [Vacuum]),
    anchor: "west",
  )
  rect((x0, 1.0), (x1, z0), fill: paper, stroke: 0.7pt + rule)
  break-band(2.75, x0, x1)
  content(
    (6.4, 3.65),
    text(size: 10pt, fill: muted, [Substrate: Si, $ε_upright(r) = 11.45$]),
    anchor: "center",
  )

  // TSV through the substrate under the left ground plane.
  rect((1.85, 1.0), (2.35, z0), fill: slate, stroke: none)
  break-band(2.75, 1.85, 2.35)

  // CPW: ground, slot, signal, slot, ground. The slots are M1_ETCH.
  let m1 = z(200)
  rect((x0, z0), (3.5, m1), fill: blue, stroke: none)
  rect((4.0, z0), (5.0, m1), fill: light-blue, stroke: 0.5pt + dark-blue)
  rect((5.5, z0), (7.2, m1), fill: blue, stroke: none)
  for (a, b) in ((3.5, 4.0), (5.0, 5.5)) {
    line((a, z0 - 0.18), (b, z0 - 0.18), stroke: 0.6pt + muted, mark: (
      start: ">",
      end: ">",
      size: 0.08,
      fill: muted,
    ))
  }
  content(
    (4.5, z0 - 0.45),
    text(size: 6.5pt, fill: muted, [M1_ETCH (1/1) slots]),
    anchor: "center",
  )

  // Airbridge over the signal line, landing on both grounds through vias.
  rect((2.75, m1), (3.15, z(300)), fill: dark-blue, stroke: none)
  rect((5.85, m1), (6.25, z(300)), fill: dark-blue, stroke: none)
  rect((2.75, z(300)), (6.25, z(500)), fill: dark-blue, stroke: none)

  // Josephson junction bridging a gap between two M1 electrodes.
  rect((7.65, z0), (8.35, m1), fill: blue, stroke: none)
  rect((7.05, z0), (7.8, z(70)), fill: copper, stroke: none)

  // NbTiN nanowire film: 15 nm, drawn slightly thicker to stay visible.
  rect((8.8, z0), (9.7, z0 + 0.07), fill: ink, stroke: none)

  // Indium bump on an M1 pad, shortened behind a break.
  rect((10.1, z0), (11.5, m1), fill: blue, stroke: none)
  rect((10.35, m1), (11.25, 7.0), fill: rule, stroke: 0.7pt + slate)
  break-band(6.2, 10.35, 11.25)

  // z axis markers.
  line((x0 - 0.15, z0), (x0, z0), stroke: 0.6pt + muted)
  content(
    (x0 - 0.2, z0),
    text(size: 6.5pt, fill: muted, [$z = 0$]),
    anchor: "east",
  )
  line((x0 - 0.15, m1), (x0, m1), stroke: 0.6pt + muted)
  content(
    (x0 - 0.2, m1 + 0.06),
    text(size: 6.5pt, fill: muted, [200 nm]),
    anchor: "east",
  )
  line((x0 - 0.15, 1.0), (x0, 1.0), stroke: 0.6pt + muted)
  content(
    (x0 - 0.2, 1.0),
    text(size: 6.5pt, fill: muted, [−500 µm]),
    anchor: "east",
  )

  // Numbered callouts tie each drawn level to the legend.
  let callout(n, at, label-at) = {
    line(at, label-at, stroke: 0.5pt + muted)
    circle(label-at, radius: 0.17, fill: white, stroke: 0.6pt + ink)
    content(
      label-at,
      text(size: 6.5pt, weight: "bold", str(n)),
      anchor: "center",
    )
  }
  let levels = (
    (blue, none, [M1_ground], [Nb · 200 nm · $z$ 0], [derived from SIM_AREA]),
    (light-blue, dark-blue, [M1], [Nb · 200 nm · $z$ 0], [M1_DRAW (1/0)]),
    (
      dark-blue,
      none,
      [Airbridge_Via],
      [Nb · 100 nm · $z$ 200 nm],
      [AB_VIA (10/1)],
    ),
    (
      dark-blue,
      none,
      [Airbridge],
      [Nb · 200 nm · $z$ 300 nm],
      [AB_DRAW (10/0)],
    ),
    (
      copper,
      none,
      [JosephsonJunction],
      [AlOx/Al · 70 nm · $z$ 0],
      [JJ_AREA (20/0)],
    ),
    (ink, none, [NbTiN], [NbTiN · 15 nm · $z$ 0], [NbTiN (25/0)]),
    (rule, slate, [IndiumBump], [In · 10 µm · $z$ 200 nm], [IND (30/0)]),
    (slate, none, [TSV], [TiN · 500 µm · $z$ −500 µm], [TSV (31/0)]),
    (paper, rule, [Substrate], [Si · 500 µm · $z$ −500 µm], [SIM_AREA (98/0)]),
    (white, muted, [Vacuum], [vacuum · 500 µm · $z$ 200 nm], [SIM_AREA (98/0)]),
  )
  callout(1, (1.9, z(100)), (1.9, 6.0))
  callout(2, (4.5, z(100)), (4.5, 6.0))
  callout(3, (6.05, z(250)), (6.6, 6.0))
  callout(4, (3.6, z(450)), (3.6, 6.6))
  callout(5, (7.3, z(35)), (7.3, 6.0))
  callout(6, (9.25, z0 + 0.035), (9.25, 6.0))
  callout(7, (10.8, 6.6), (10.8, 7.45))
  callout(8, (2.1, 1.6), (2.75, 1.6))
  callout(9, (9.0, 1.6), (9.6, 1.6))
  callout(10, (8.0, 8.45), (8.6, 8.45))

  // Legend.
  let lx = 12.4
  for (i, (fill, stroke, name, spec, gds)) in levels.enumerate() {
    let y = 8.75 - i * 0.79
    circle((lx, y), radius: 0.17, fill: white, stroke: 0.6pt + ink)
    content(
      (lx, y),
      text(size: 6.5pt, weight: "bold", str(i + 1)),
      anchor: "center",
    )
    rect(
      (lx + 0.35, y - 0.17),
      (lx + 0.75, y + 0.17),
      fill: fill,
      stroke: if stroke == none { none } else { 0.6pt + stroke },
    )
    content(
      (lx + 0.95, y + 0.14),
      text(size: 7.5pt, font: heading-font, weight: "bold", name),
      anchor: "west",
    )
    content(
      (lx + 0.95, y - 0.18),
      text(size: 6.5pt, fill: muted, [#spec · #gds]),
      anchor: "west",
    )
  }
})
