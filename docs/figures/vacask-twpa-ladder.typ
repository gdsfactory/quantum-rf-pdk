#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 158mm, height: 46mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let label(pos, body, fill: ink, size: 9pt, anchor: "center") = content(
    pos,
    text(size: size, font: heading-font, weight: "bold", fill: fill, body),
    anchor: anchor,
  )
  let ground(x, y) = {
    line((x - 0.32, y), (x + 0.32, y), stroke: 1.3pt + slate)
    line((x - 0.2, y - 0.1), (x + 0.2, y - 0.1), stroke: 1.3pt + slate)
    line((x - 0.08, y - 0.2), (x + 0.08, y - 0.2), stroke: 1.3pt + slate)
  }
  let resistor(x, top, bottom) = {
    let mid = (top + bottom) / 2
    line((x, top), (x, mid + 0.4), stroke: 1.5pt + slate)
    rect((x - 0.17, mid + 0.4), (x + 0.17, mid - 0.4), stroke: 1.5pt + slate)
    line((x, mid - 0.4), (x, bottom), stroke: 1.5pt + slate)
  }
  // One cell: series junction from x to x + 1.6, shunt capacitor at x + 1.6
  let cell(x, y) = {
    line((x, y), (x + 0.5, y), stroke: 1.5pt + ink)
    line((x + 0.5, y + 0.28), (x + 1.06, y - 0.28), stroke: 1.7pt + dark-blue)
    line((x + 0.5, y - 0.28), (x + 1.06, y + 0.28), stroke: 1.7pt + dark-blue)
    line((x + 0.5, y), (x + 1.06, y), stroke: 1.5pt + dark-blue)
    line((x + 1.06, y), (x + 1.6, y), stroke: 1.5pt + ink)
    circle((x + 1.6, y), radius: 0.06, fill: ink, stroke: none)
    line((x + 1.6, y), (x + 1.6, y - 0.62), stroke: 1.5pt + light-blue)
    line((x + 1.25, y - 0.62), (x + 1.95, y - 0.62), stroke: 2pt + light-blue)
    line((x + 1.25, y - 0.84), (x + 1.95, y - 0.84), stroke: 2pt + light-blue)
    line((x + 1.6, y - 0.84), (x + 1.6, y - 1.35), stroke: 1.5pt + light-blue)
    ground(x + 1.6, y - 1.35)
  }

  content(
    (0.5, 4.15),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [JOSEPHSON-JUNCTION LADDER TWPA],
    ),
    anchor: "west",
  )
  content(
    (0.5, 3.72),
    text(
      size: 8pt,
      fill: muted,
      [Each cell is a series junction and a shunt capacitor; $sqrt(L_upright(J) \/ C) ≈ 50$ Ω.],
    ),
    anchor: "west",
  )

  let y = 2.55
  // source port
  circle((1.2, y), radius: 0.11, fill: blue, stroke: none)
  label((1.2, y + 0.42), [pump + signal], fill: blue)
  resistor(1.2, y, y - 1.35)
  ground(1.2, y - 1.35)
  label((0.8, y - 0.68), [50 Ω], fill: slate, anchor: "east")
  line((1.2, y), (1.6, y), stroke: 1.5pt + ink)

  cell(1.6, y)
  cell(3.2, y)
  cell(4.8, y)
  label((5.6, y + 0.5), $I_upright(c)$, fill: dark-blue)
  label((6.95, y - 0.73), $C$, fill: light-blue, anchor: "west")
  line((6.4, y), (6.9, y), stroke: 1.5pt + ink)
  label((7.6, y), [⋯], fill: ink)
  line((8.3, y), (8.6, y), stroke: 1.5pt + ink)
  cell(8.6, y)
  line((10.2, y), (11.2, y), stroke: 1.5pt + ink)

  // load
  circle((11.2, y), radius: 0.11, fill: blue, stroke: none)
  label((11.2, y + 0.42), [output], fill: blue)
  resistor(11.2, y, y - 1.35)
  ground(11.2, y - 1.35)
  label((11.6, y - 0.68), [50 Ω], fill: slate, anchor: "west")

  // brace for N cells
  line((1.6, 0.75), (10.2, 0.75), stroke: 0.8pt + muted)
  line((1.6, 0.65), (1.6, 0.85), stroke: 0.8pt + muted)
  line((10.2, 0.65), (10.2, 0.85), stroke: 0.8pt + muted)
  content(
    (5.9, 0.4),
    text(size: 8pt, fill: muted, [$N$ = 200 … 2000 cells]),
    anchor: "center",
  )
})
