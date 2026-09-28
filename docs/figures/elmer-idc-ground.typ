#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper,
)

#set page(width: 128mm, height: 115mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

// Top view of the IDC electrostatic model. Distances are schematic, not to scale:
// the 10 µm etch, 45 µm ground edge and 90 µm domain are drawn with even spacing.
#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let xs = range(8).map(i => 4.30 + i * 0.56 + (if i >= 4 { 0.5 } else { 0 }))

  rect(
    (1.60, 0.85),
    (11.64, 9.55),
    fill: paper,
    stroke: (paint: muted, thickness: 0.6pt, dash: "dashed"),
  )
  rect((2.55, 1.80), (10.69, 8.60), fill: blue, stroke: none)
  rect((3.50, 2.75), (9.74, 7.65), fill: paper, stroke: none)

  rect((4.05, 6.60), (9.19, 7.10), fill: light-blue, stroke: 0.4pt + blue)
  rect((4.05, 3.30), (9.19, 3.80), fill: light-blue, stroke: 0.4pt + blue)
  for (i, x) in xs.enumerate() {
    let from-top = calc.even(i)
    rect(
      (x, if from-top { 4.10 } else { 3.80 }),
      (x + 0.22, if from-top { 6.60 } else { 6.30 }),
      fill: light-blue,
      stroke: 0.4pt + blue,
    )
  }

  content(
    (4.25, 6.85),
    text(size: 7pt, fill: ink, [terminal $o_1$]),
    anchor: "west",
  )
  content(
    (4.25, 3.55),
    text(size: 7pt, fill: ink, [terminal $o_2$]),
    anchor: "west",
  )
  content(
    (6.62, 8.12),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: paper,
      [grounded M1 frame],
    ),
    anchor: "center",
  )
  content(
    (7.20, 7.375),
    text(size: 7pt, fill: muted, [etched clearance]),
    anchor: "center",
  )
  content(
    (11.45, 9.20),
    text(size: 7pt, fill: muted, [dielectric domain]),
    anchor: "east",
  )

  line((6.20, 5.30), (6.48, 5.30), stroke: 0.8pt + dark-blue)
  line((6.48, 4.95), (6.48, 5.65), stroke: 1.2pt + dark-blue)
  line((6.76, 4.95), (6.76, 5.65), stroke: 1.2pt + dark-blue)
  line((6.76, 5.30), (7.04, 5.30), stroke: 0.8pt + dark-blue)
  content(
    (6.62, 4.55),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      $-C_12$,
    ),
    anchor: "center",
  )

  line((4.90, 7.10), (4.90, 7.25), stroke: 0.8pt + dark-blue)
  line((4.72, 7.25), (5.08, 7.25), stroke: 1.2pt + dark-blue)
  line((4.72, 7.45), (5.08, 7.45), stroke: 1.2pt + dark-blue)
  line((4.90, 7.45), (4.90, 7.65), stroke: 0.8pt + dark-blue)
  content(
    (5.20, 7.35),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      $C_(1 upright(g))$,
    ),
    anchor: "west",
  )
  line((8.30, 3.30), (8.30, 3.15), stroke: 0.8pt + dark-blue)
  line((8.12, 3.15), (8.48, 3.15), stroke: 1.2pt + dark-blue)
  line((8.12, 2.95), (8.48, 2.95), stroke: 1.2pt + dark-blue)
  line((8.30, 2.95), (8.30, 2.75), stroke: 0.8pt + dark-blue)
  content(
    (8.60, 3.05),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      $C_(2 upright(g))$,
    ),
    anchor: "west",
  )

  line((9.19, 3.60), (9.19, 5.85), stroke: (
    paint: muted,
    thickness: 0.5pt,
    dash: "dotted",
  ))
  line((9.74, 5.40), (9.74, 5.80), stroke: (
    paint: muted,
    thickness: 0.5pt,
    dash: "dotted",
  ))
  line((10.69, 4.50), (10.69, 4.90), stroke: (
    paint: muted,
    thickness: 0.5pt,
    dash: "dotted",
  ))
  line(
    (9.19, 5.60),
    (9.74, 5.60),
    stroke: 0.8pt + ink,
    mark: (start: ">", end: ">", size: 0.11, fill: ink),
  )
  line(
    (9.19, 4.70),
    (10.69, 4.70),
    stroke: 0.8pt + ink,
    mark: (start: ">", end: ">", size: 0.11, fill: ink),
  )
  line(
    (9.19, 3.80),
    (11.64, 3.80),
    stroke: 0.8pt + ink,
    mark: (start: ">", end: ">", size: 0.11, fill: ink),
  )
  for (cx, cy) in ((9.94, 4.70), (10.415, 3.80)) {
    rect(
      (cx - 0.45, cy - 0.10),
      (cx + 0.45, cy + 0.28),
      fill: paper,
      stroke: none,
    )
  }
  content(
    (9.465, 5.94),
    text(size: 6pt, fill: ink, [10 µm]),
    anchor: "center",
  )
  content(
    (9.94, 4.79),
    text(size: 7pt, fill: ink, [45 µm]),
    anchor: "center",
  )
  content(
    (10.415, 3.89),
    text(size: 7pt, fill: ink, [90 µm]),
    anchor: "center",
  )

  content(
    (1.75, 1.30),
    text(
      size: 7pt,
      fill: muted,
      [Schematic, not to scale. Distances from the comb bounding box edge.],
    ),
    anchor: "west",
  )

  content(
    (1.60, 10.45),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [IDC ELECTROSTATIC MODEL],
    ),
    anchor: "west",
  )
  content(
    (1.60, 10.15),
    text(
      size: 8pt,
      fill: muted,
      [Top view of the two isolated combs and their grounded coplanar frame.],
    ),
    anchor: "west",
  )
  content(
    (1.60, 9.85),
    text(
      size: 8pt,
      fill: muted,
      [Couplings: $-C_12$ mutual, $C_(1 upright(g))$ / $C_(2 upright(g))$ to ground.],
    ),
    anchor: "west",
  )
})
