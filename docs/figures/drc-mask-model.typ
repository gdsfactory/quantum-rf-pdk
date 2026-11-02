#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
)

#set page(width: 158mm, height: 76mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let copper = rgb("#a95a20")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 7.15),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [THE MASK MODEL IS A SUBTRACTION],
    ),
    anchor: "west",
  )
  content(
    (0.5, 6.7),
    text(
      size: 8pt,
      fill: muted,
      [The film is drawn blanket and etched back, and the rules measure what remains.],
    ),
    anchor: "west",
  )

  let x0 = 4.2
  let x1 = 15.2
  let slot-a = (6.4, 8.2)
  let slot-b = (11.2, 13.0)

  // The film as drawn.
  content(
    (x0 - 0.2, 5.875),
    text(size: 8pt, fill: muted, [M1_DRAW]),
    anchor: "east",
  )
  rect((x0, 5.55), (x1, 6.2), fill: light-blue, stroke: 0.6pt + blue)
  content(
    ((x0 + x1) / 2, 5.875),
    text(size: 7.5pt, fill: ink, [blanket niobium]),
    anchor: "center",
  )

  // The etch mask.
  content(
    (x0 - 0.2, 4.775),
    text(size: 8pt, fill: muted, [M1_ETCH]),
    anchor: "east",
  )
  rect((x0, 4.45), (x1, 5.1), fill: paper, stroke: 0.6pt + rule)
  rect((slot-a.at(0), 4.45), (slot-a.at(1), 5.1), fill: copper, stroke: none)
  rect((slot-b.at(0), 4.45), (slot-b.at(1), 5.1), fill: copper, stroke: none)
  content(
    ((slot-a.at(0) + slot-a.at(1)) / 2, 4.775),
    text(size: 7.5pt, fill: white, [etch]),
    anchor: "center",
  )
  content(
    ((slot-b.at(0) + slot-b.at(1)) / 2, 4.775),
    text(size: 7.5pt, fill: white, [etch]),
    anchor: "center",
  )

  // What the film becomes.
  content(
    (x0 - 0.2, 3.475),
    text(size: 8pt, fill: muted, [M1 metal]),
    anchor: "east",
  )
  rect((x0, 3.15), (slot-a.at(0), 3.8), fill: light-blue, stroke: none)
  rect(
    (slot-a.at(1), 3.15),
    (slot-b.at(0), 3.8),
    fill: light-blue,
    stroke: none,
  )
  rect((slot-b.at(1), 3.15), (x1, 3.8), fill: light-blue, stroke: none)

  let ground-a = (x0 + slot-a.at(0)) / 2
  let signal = (slot-a.at(1) + slot-b.at(0)) / 2
  let ground-b = (slot-b.at(1) + x1) / 2
  content(
    (ground-a, 2.8),
    text(size: 7.5pt, fill: muted, [ground]),
    anchor: "center",
  )
  content(
    (signal, 2.8),
    text(size: 7.5pt, fill: muted, [signal]),
    anchor: "center",
  )
  content(
    (ground-b, 2.8),
    text(size: 7.5pt, fill: muted, [ground]),
    anchor: "center",
  )

  // Width is measured on the film that remains.
  line(
    (slot-a.at(1), 2.15),
    (slot-b.at(0), 2.15),
    stroke: 0.8pt + dark-blue,
    mark: (start: ">", end: ">", size: 0.12, fill: dark-blue),
  )
  content(
    (signal, 1.8),
    text(
      size: 7.5pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [$w$],
    ),
    anchor: "center",
  )

  // Spacing is the width of the etched slot.
  line(
    (slot-b.at(0), 4.18),
    (slot-b.at(1), 4.18),
    stroke: 0.8pt + dark-blue,
    mark: (start: ">", end: ">", size: 0.12, fill: dark-blue),
  )
  content(
    ((slot-b.at(0) + slot-b.at(1)) / 2, 3.97),
    text(
      size: 7.5pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [$s$],
    ),
    anchor: "center",
  )
})
