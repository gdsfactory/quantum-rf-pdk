#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 190mm, height: 178mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let copper = rgb("#a95a20")

// Cross-sections of the QPDK layer stacks. Not to scale: the main panel draws
// `LAYER_STACK` plus the `LAYER_STACK_FLIP_CHIP` top chip with the thin films at
// 1 cm = 400 nm, while the 500 µm substrates, the vacuum and the 10 µm indium
// bump are shortened behind break marks. The right panel draws the
// `LAYER_STACK_GATEMON` films at 1 cm = 50 nm.
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let nm = 1 / 400
  let z0 = 11.2
  let z(v) = z0 + v * nm
  let x0 = 1.3
  let x1 = 11.8
  let top = 15.45

  content(
    (0.5, 16.55),
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
    (0.5, 16.1),
    text(
      size: 8pt,
      fill: muted,
      [Cross-sections, not to scale. Every level of every stack is numbered and listed below.],
    ),
    anchor: "west",
  )

  let break-band(y, from, to) = {
    rect((from, y - 0.13), (to, y + 0.13), fill: white, stroke: none)
    let steps = calc.max(1, int((to - from) / 0.3))
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

  let small(body) = text(size: 6.5pt, fill: muted, body)

  // Vacuum above the metal and the silicon substrate below it.
  rect(
    (x0, z(200)),
    (x1, top),
    fill: white,
    stroke: (paint: muted, thickness: 0.6pt, dash: "dashed"),
  )
  break-band(14.45, x0, 6.6)
  content(
    (x0 + 0.25, 15.05),
    text(size: 8pt, fill: muted, [Vacuum]),
    anchor: "west",
  )
  rect((x0, 7.6), (x1, z0), fill: paper, stroke: 0.7pt + rule)
  break-band(9.35, x0, x1)
  content(
    (6.4, 10.25),
    text(size: 10pt, fill: muted, [Substrate: Si, $ε_upright(r) = 11.45$]),
    anchor: "center",
  )

  // TSV through the substrate under the left ground plane.
  rect((1.85, 7.6), (2.35, z0), fill: slate, stroke: none)
  break-band(9.35, 1.85, 2.35)

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
  content((4.5, z0 - 0.45), small[M1_ETCH (1/1) slots], anchor: "center")

  // Airbridge over the signal line, landing on both grounds through vias.
  rect((2.75, m1), (3.15, z(300)), fill: dark-blue, stroke: none)
  rect((5.85, m1), (6.25, z(300)), fill: dark-blue, stroke: none)
  rect((2.75, z(300)), (6.25, z(500)), fill: dark-blue, stroke: none)

  // Josephson junction bridging a gap between two M1 electrodes.
  rect((7.65, z0), (8.35, m1), fill: blue, stroke: none)
  rect((7.05, z0), (7.8, z(70)), fill: copper, stroke: none)

  // NbTiN nanowire film: 15 nm, drawn slightly thicker to stay visible.
  rect((8.8, z0), (9.7, z0 + 0.07), fill: ink, stroke: none)

  // Flip-chip top chip (LAYER_STACK_FLIP_CHIP): M2 faces down at z = 10 µm,
  // carried by a second substrate. The indium bump joins the M1 and M2 pads.
  let zt = 13.6
  let m2 = zt + 0.5
  rect((6.6, m2), (x1, top), fill: paper, stroke: 0.7pt + rule)
  break-band(14.9, 6.6, x1)
  rect((6.6, zt), (9.75, m2), fill: blue, stroke: none)
  rect((10.1, zt), (x1, m2), fill: light-blue, stroke: 0.5pt + dark-blue)
  content((9.88, zt - 0.2), small[M2_ETCH (2/1)], anchor: "north-east")
  rect((10.1, z0), (11.5, m1), fill: blue, stroke: none)
  rect((10.35, m1), (11.25, zt), fill: rule, stroke: 0.7pt + slate)
  break-band(12.6, 10.35, 11.25)

  // z axis markers.
  let tick(y, label, side: "left") = {
    if side == "left" {
      line((x0 - 0.15, y), (x0, y), stroke: 0.6pt + muted)
      content((x0 - 0.2, y), small(label), anchor: "east")
    } else {
      line((x1, y), (x1 + 0.15, y), stroke: 0.6pt + muted)
      content((x1 + 0.2, y), small(label), anchor: "west")
    }
  }
  tick(z0, [$z = 0$])
  tick(m1 + 0.06, [200 nm])
  tick(7.6, [−500 µm])
  tick(zt, [10 µm], side: "right")
  tick(m2, [10.2 µm], side: "right")

  // Gatemon panel: z = 0 is the oxide surface, 1 cm = 50 nm.
  let gx0 = 13.3
  let gx1 = 18.4
  let g(v) = z0 + v / 50
  content(
    ((gx0 + gx1) / 2, 15.25),
    text(size: 7pt, font: heading-font, weight: "bold", [Gatemon films]),
    anchor: "center",
  )
  content(
    ((gx0 + gx1) / 2, 14.9),
    small[$z = 0$ at the oxide surface],
    anchor: "center",
  )
  rect(
    (gx0, g(0)),
    (gx1, 14.65),
    fill: white,
    stroke: (paint: muted, thickness: 0.6pt, dash: "dashed"),
  )
  rect((gx0, 7.6), (gx1, g(-150)), fill: paper, stroke: 0.7pt + rule)
  break-band(7.9, gx0, gx1)
  rect((gx0, g(-150)), (gx1, g(0)), fill: rule, stroke: none)
  // Nanotube between the contacts: not extruded, shown for orientation.
  line((14.6, g(0) + 0.03), (17.1, g(0) + 0.03), stroke: (
    paint: ink,
    thickness: 0.8pt,
    dash: "dotted",
  ))
  content(
    (15.85, g(0) - 0.25),
    small[nanotube (not extruded)],
    anchor: "center",
  )
  for (cx0, cx1) in ((14.2, 15.0), (16.7, 17.5)) {
    rect((cx0, g(0)), (cx1, g(35)), fill: dark-blue, stroke: none)
    rect((cx0, g(35)), (cx1, g(45)), fill: copper, stroke: none)
  }
  for (rx0, rx1) in ((13.6, 14.6), (17.1, 18.1)) {
    rect((rx0, g(45)), (rx1, g(145)), fill: slate, stroke: none)
  }
  for (mx0, mx1) in ((gx0, 13.85), (17.85, gx1)) {
    rect(
      (mx0, g(0)),
      (mx1, g(155)),
      fill: light-blue,
      stroke: 0.5pt + dark-blue,
    )
  }
  rect((14.85, g(45)), (16.85, g(75)), fill: paper, stroke: 0.6pt + slate)
  rect((15.35, g(75)), (16.35, g(175)), fill: slate, stroke: none)

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
  callout(1, (1.9, z(100)), (1.9, 12.6))
  callout(2, (4.5, z(100)), (4.5, 12.6))
  callout(3, (6.05, z(250)), (6.6, 12.6))
  callout(4, (3.6, z(450)), (3.6, 13.2))
  callout(5, (7.3, z(35)), (7.3, 12.6))
  callout(6, (9.25, z0 + 0.035), (9.25, 12.6))
  callout(7, (10.35, 12.1), (9.85, 12.1))
  callout(8, (2.1, 8.2), (2.75, 8.2))
  callout(9, (9.0, 8.2), (9.6, 8.2))
  callout(10, (3.1, 15.05), (3.1, 15.05))
  callout(11, (10.9, zt + 0.25), (10.9, 14.4))
  callout(12, (8.0, zt + 0.25), (8.0, 14.4))
  callout(13, (9.4, 15.22), (9.4, 15.22))
  callout(2, (13.6, g(120)), (13.6, 14.3))
  callout(14, (15.85, g(-80)), (15.85, g(-80)))
  callout(15, (14.35, g(17)), (14.35, 9.9))
  callout(16, (17.35, g(40)), (17.35, 9.9))
  callout(17, (16.6, g(60)), (16.95, 14.3))
  callout(18, (15.85, g(125)), (15.85, g(125)))
  callout(19, (17.6, g(110)), (17.6, 14.3))

  // Legend, grouped by stack.
  let level(fill, stroke, name, spec, gds) = (fill, stroke, name, spec, gds)
  let base = (
    level(
      blue,
      none,
      [M1_ground],
      [Nb · 200 nm · $z$ 0],
      [derived from SIM_AREA],
    ),
    level(light-blue, dark-blue, [M1], [Nb · 200 nm · $z$ 0], [M1_DRAW (1/0)]),
    level(
      dark-blue,
      none,
      [Airbridge_Via],
      [Nb · 100 nm · $z$ 200 nm],
      [AB_VIA (10/1)],
    ),
    level(
      dark-blue,
      none,
      [Airbridge],
      [Nb · 200 nm · $z$ 300 nm],
      [AB_DRAW (10/0)],
    ),
    level(
      copper,
      none,
      [JosephsonJunction],
      [AlOx/Al · 70 nm · $z$ 0],
      [JJ_AREA (20/0)],
    ),
    level(ink, none, [NbTiN], [NbTiN · 15 nm · $z$ 0], [NbTiN (25/0)]),
    level(rule, slate, [IndiumBump], [In · 10 µm · $z$ 200 nm], [IND (30/0)]),
    level(slate, none, [TSV], [TiN · 500 µm · $z$ −500 µm], [TSV (31/0)]),
    level(
      paper,
      rule,
      [Substrate],
      [Si · 500 µm · $z$ −500 µm],
      [SIM_AREA (98/0)],
    ),
    level(
      white,
      muted,
      [Vacuum],
      [vacuum · 500 µm · $z$ 200 nm],
      [SIM_AREA (98/0)],
    ),
  )
  let flip-chip = (
    level(
      light-blue,
      dark-blue,
      [M2],
      [Nb · 200 nm · $z$ 10 µm],
      [M2_DRAW (2/0)],
    ),
    level(
      blue,
      none,
      [M2_ground],
      [Nb · 200 nm · $z$ 10 µm],
      [derived from SIM_AREA],
    ),
    level(
      paper,
      rule,
      [Substrate_top],
      [Si · 500 µm · $z$ 10.2 µm],
      [SIM_AREA (98/0)],
    ),
  )
  let gatemon = (
    level(
      rule,
      none,
      [Oxide],
      [SiO#sub[2] · 150 nm · $z$ −150 nm],
      [SIM_AREA (98/0)],
    ),
    level(
      dark-blue,
      none,
      [ContactNb],
      [Nb · 35 nm · $z$ 0],
      [CNT_CONTACT (21/0)],
    ),
    level(
      copper,
      none,
      [ContactAu],
      [Au · 10 nm · $z$ 35 nm],
      [CNT_CONTACT (21/0)],
    ),
    level(
      paper,
      slate,
      [hBN],
      [hBN · 30 nm (default) · $z$ 45 nm],
      [HBN (23/0)],
    ),
    level(slate, none, [TopGate], [Al · 100 nm · $z$ 75 nm], [TOP_GATE (22/0)]),
    level(
      slate,
      none,
      [Reconnection],
      [Al · 100 nm · $z$ 45 nm],
      [JJ_PATCH (20/1)],
    ),
  )

  let row-h = 0.7
  let heading(x, y, body) = content(
    (x, y),
    text(size: 7pt, font: heading-font, weight: "bold", fill: blue, body),
    anchor: "west",
  )
  let row(x, y, n, entry) = {
    let (fill, stroke, name, spec, gds) = entry
    circle((x + 0.17, y), radius: 0.17, fill: white, stroke: 0.6pt + ink)
    content(
      (x + 0.17, y),
      text(size: 6.5pt, weight: "bold", str(n)),
      anchor: "center",
    )
    rect(
      (x + 0.52, y - 0.17),
      (x + 0.92, y + 0.17),
      fill: fill,
      stroke: if stroke == none { none } else { 0.6pt + stroke },
    )
    content(
      (x + 1.12, y + 0.14),
      text(size: 7.5pt, font: heading-font, weight: "bold", name),
      anchor: "west",
    )
    content((x + 1.12, y - 0.18), small[#spec · #gds], anchor: "west")
  }

  let ly = 6.95
  heading(0.5, ly, [LAYER_STACK (base process, all stacks)])
  for (i, entry) in base.enumerate() {
    row(0.5, ly - 0.55 - i * row-h, i + 1, entry)
  }
  let cx = 9.9
  heading(cx, ly, [LAYER_STACK_FLIP_CHIP adds the top chip])
  for (i, entry) in flip-chip.enumerate() {
    row(cx, ly - 0.55 - i * row-h, i + 11, entry)
  }
  let gy = ly - 0.55 - 3 * row-h - 0.15
  heading(cx, gy, [LAYER_STACK_GATEMON adds (M1 is 155 nm thick here)])
  for (i, entry) in gatemon.enumerate() {
    row(cx, gy - 0.55 - i * row-h, i + 14, entry)
  }
})
