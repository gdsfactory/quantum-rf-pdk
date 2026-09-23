#import "@preview/cetz:0.4.2"
#import "style.typ": blue, coral, cyan, ink, muted, paper, purple, rule

#set page(width: 156mm, height: 69mm, margin: 3mm, fill: white)
#set text(font: "Arial", fill: ink)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.55, 5.35),
    text(
      size: 8pt,
      weight: "bold",
      fill: blue,
      [CAPACITIVELY COUPLED QUARTER-WAVE RESONATOR],
    ),
    anchor: "west",
  )
  content(
    (0.55, 4.9),
    text(
      size: 8pt,
      fill: muted,
      [A shunt branch adds a narrow transmission dip to the feedline.],
    ),
    anchor: "west",
  )

  line((1.1, 3.65), (13.9, 3.65), stroke: 2.2pt + blue)
  circle((7.5, 3.65), radius: 0.11, fill: blue, stroke: none)
  content(
    (1.1, 4.05),
    text(size: 9pt, weight: "bold", fill: blue, [port 1]),
    anchor: "west",
  )
  content(
    (13.9, 4.05),
    text(size: 9pt, weight: "bold", fill: blue, [port 2]),
    anchor: "east",
  )
  content(
    (11.7, 3.25),
    text(size: 8pt, fill: muted, [through feedline]),
    anchor: "center",
  )

  line((7.5, 3.65), (7.5, 3.04), stroke: 1.6pt + cyan)
  line((7.05, 3.04), (7.95, 3.04), stroke: 2pt + cyan)
  line((7.05, 2.8), (7.95, 2.8), stroke: 2pt + cyan)
  content(
    (8.2, 2.92),
    text(size: 9pt, weight: "bold", fill: cyan, $C_c$),
    anchor: "west",
  )

  line((7.5, 2.8), (7.5, 0.95), stroke: 3pt + purple)
  content(
    (8.2, 1.9),
    text(size: 10pt, weight: "bold", fill: purple, [λ / 4 resonator]),
    anchor: "west",
  )
  content(
    (8.2, 1.48),
    text(size: 8pt, fill: muted, [coupled end near the feedline]),
    anchor: "west",
  )
  line((7.5, 0.95), (7.5, 0.7), stroke: 1.6pt + coral)
  line((6.95, 0.7), (8.05, 0.7), stroke: 1.3pt + coral)
  line((7.12, 0.57), (7.88, 0.57), stroke: 1.3pt + coral)
  line((7.3, 0.44), (7.7, 0.44), stroke: 1.3pt + coral)
  content(
    (6.55, 0.68),
    text(size: 8pt, weight: "bold", fill: coral, [short]),
    anchor: "east",
  )
})
