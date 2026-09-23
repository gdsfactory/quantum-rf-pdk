#import "@preview/cetz:0.4.2"
#import "style.typ": blue, coral, cyan, ink, muted, paper, purple, rule

#set page(width: 162mm, height: 55mm, margin: 3mm, fill: white)
#set text(font: "Arial", fill: ink)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.25),
    text(
      size: 8pt,
      weight: "bold",
      fill: blue,
      [WHERE SAX STOPS EXPANDING A CHIP],
    ),
    anchor: "west",
  )
  content(
    (0.5, 4.82),
    text(
      size: 8pt,
      fill: muted,
      [Unmodeled assemblies expand; cells with a compact model stay intact.],
    ),
    anchor: "west",
  )

  rect(
    (0.55, 0.72),
    (5.65, 4.3),
    radius: 0.16,
    fill: paper,
    stroke: 0.8pt + rule,
  )
  content(
    (0.85, 3.87),
    text(size: 10pt, weight: "bold", fill: ink, [chip assembly]),
    anchor: "west",
  )
  content(
    (4.9, 3.87),
    text(size: 7.5pt, fill: muted, [no model]),
    anchor: "east",
  )
  rect(
    (0.9, 2.55),
    (5.3, 3.48),
    radius: 0.12,
    fill: blue.transparentize(87%),
    stroke: 0.8pt + blue,
  )
  content(
    (1.2, 3.04),
    text(size: 9pt, fill: blue, [coupled resonator]),
    anchor: "west",
  )
  rect(
    (0.9, 1.39),
    (5.3, 2.31),
    radius: 0.12,
    fill: cyan.transparentize(85%),
    stroke: 0.8pt + cyan,
  )
  content((1.2, 1.88), text(size: 9pt, fill: cyan, [CPW route]), anchor: "west")

  line((5.9, 2.5), (8.0, 2.5), stroke: 1.2pt + blue, mark: (
    end: ">",
    size: 0.18,
    fill: blue,
  ))
  content(
    (6.95, 2.91),
    text(size: 8pt, fill: muted, [expand]),
    anchor: "center",
  )

  rect(
    (8.3, 2.55),
    (15.1, 4.16),
    radius: 0.13,
    fill: blue.transparentize(87%),
    stroke: 0.8pt + blue,
  )
  content(
    (8.65, 3.62),
    text(size: 10pt, weight: "bold", fill: blue, [coupled resonator model]),
    anchor: "west",
  )
  content(
    (8.65, 3.05),
    text(size: 8pt, fill: muted, [coupling preserved inside this boundary]),
    anchor: "west",
  )
  rect(
    (8.3, 0.83),
    (15.1, 2.35),
    radius: 0.13,
    fill: cyan.transparentize(85%),
    stroke: 0.8pt + cyan,
  )
  content(
    (8.65, 1.81),
    text(size: 10pt, weight: "bold", fill: cyan, [modeled CPW primitives]),
    anchor: "west",
  )
  content(
    (8.65, 1.24),
    text(size: 8pt, fill: muted, [straights, bends, and other modeled cells]),
    anchor: "west",
  )
})
