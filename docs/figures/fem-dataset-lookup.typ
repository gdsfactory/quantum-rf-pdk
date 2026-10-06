#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
)

#set page(width: 140mm, height: 80mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let xs = (1.5, 2.5, 4.0, 5.5, 7.0, 9.0)
#let ys = (1.2, 2.4, 3.9, 5.4)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 7.3),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [MULTILINEAR LOOKUP OF A FEM DATASET],
    ),
    anchor: "west",
  )
  content(
    (0.5, 6.85),
    text(
      size: 8pt,
      fill: muted,
      [Solved points span the grid; a query blends the corners of its cell.],
    ),
    anchor: "west",
  )

  rect(
    (xs.first(), ys.first()),
    (xs.last(), ys.last()),
    fill: paper,
    stroke: 0.7pt + rule,
  )
  for x in xs { line((x, ys.first()), (x, ys.last()), stroke: 0.5pt + rule) }
  for y in ys { line((xs.first(), y), (xs.last(), y), stroke: 0.5pt + rule) }

  rect(
    (4.0, 2.4),
    (5.5, 3.9),
    fill: light-blue.transparentize(80%),
    stroke: none,
  )
  let q = (4.45, 3.15)
  for corner in ((4.0, 2.4), (5.5, 2.4), (4.0, 3.9), (5.5, 3.9)) {
    line(q, corner, stroke: (
      paint: dark-blue,
      thickness: 0.6pt,
      dash: "dashed",
    ))
  }
  for x in xs {
    for y in ys { circle((x, y), radius: 0.07, fill: blue, stroke: none) }
  }
  circle(q, radius: 0.11, fill: dark-blue, stroke: none)
  content(
    (4.62, 3.15),
    text(
      size: 7pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [query],
    ),
    anchor: "west",
  )

  circle((9.75, 3.15), radius: 0.11, fill: white, stroke: 0.8pt + ink)
  content(
    (9.75, 2.85),
    text(size: 8pt, font: heading-font, weight: "bold", [NaN]),
    anchor: "north",
  )

  content(
    (5.25, 0.75),
    text(size: 8pt, [length $l$ ($upright(µ m)$)]),
    anchor: "center",
  )
  content(
    (1.35, 5.75),
    text(size: 8pt, [gap $s$ ($upright(µ m)$)]),
    anchor: "west",
  )

  content(
    (10.6, 5.4),
    box(width: 3.1cm, text(size: 8pt, [
      #text(fill: blue, font: heading-font, weight: "bold")[solved point] \
      One FEM run: a full $C$ matrix.

      #text(fill: dark-blue, font: heading-font, weight: "bold")[query] \
      Weighted mean of the $2^n$ corners of its cell.

      #text(font: heading-font, weight: "bold")[outside] \
      No data, so NaN rather than a clamped edge value.
    ])),
    anchor: "north-west",
  )
})
