#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
)

#set page(width: 158mm, height: 48mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

// The two readings of one process: the films in section and the same films as
// masks, which is what the layer stack and the design rules pages each show.
// Deliberately schematic: the numbers live on the layer stack page.
#let copper = rgb("#a95a20")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  rect((0.1, 0.3), (7.3, 4.45), stroke: 0.6pt + rule)
  rect((7.9, 0.3), (15.1, 4.45), stroke: 0.6pt + rule)

  // ── Left: the films in section ────────────────────────────────────────────
  content(
    (0.7, 4.15),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [LAYER STACK],
    ),
    anchor: "west",
  )
  content(
    (0.7, 3.85),
    text(size: 7.5pt, fill: muted, [a section through the films]),
    anchor: "west",
  )

  rect((1.5, 1.15), (4.1, 2.2), fill: paper, stroke: 0.7pt + rule)
  content(
    (2.8, 1.67),
    text(size: 7pt, fill: muted, [silicon]),
    anchor: "center",
  )
  rect((1.5, 2.2), (4.1, 2.5), fill: light-blue, stroke: 0.6pt + blue)
  content(
    (2.8, 2.35),
    text(size: 6.5pt, fill: ink, [Nb]),
    anchor: "center",
  )
  rect((1.85, 2.5), (2.45, 2.65), fill: copper, stroke: none)
  line((2.45, 2.57), (3.15, 3.05), stroke: 0.6pt + muted)
  content(
    (3.2, 3.1),
    text(size: 7pt, fill: muted, [Al/AlOx junction]),
    anchor: "west",
  )
  line((4.1, 2.35), (4.85, 2.35), stroke: 0.6pt + muted)
  content(
    (4.9, 2.35),
    text(size: 7pt, fill: muted, [niobium M1]),
    anchor: "west",
  )
  content(
    (3.7, 0.6),
    text(size: 7pt, fill: muted, [films, materials, thicknesses, GDS layers]),
    anchor: "center",
  )

  // ── Right: the same films from above ──────────────────────────────────────
  content(
    (8.5, 4.15),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [DESIGN RULES],
    ),
    anchor: "west",
  )
  content(
    (8.5, 3.85),
    text(size: 7.5pt, fill: muted, [the same films as masks, from above]),
    anchor: "west",
  )

  let m1 = (y0: 1.5, y1: 2.75)
  let ground-a = (8.2, 9.8)
  let slot-a = (9.8, 10.2)
  let signal = (10.2, 12.0)
  let ground-b = (12.4, 14.0)

  for strip in (ground-a, signal, ground-b) {
    rect(
      (strip.at(0), m1.y0),
      (strip.at(1), m1.y1),
      fill: light-blue,
      stroke: none,
    )
  }
  content(
    ((ground-a.at(0) + ground-a.at(1)) / 2, 2.12),
    text(size: 7pt, fill: muted, [ground]),
    anchor: "center",
  )
  content(
    ((signal.at(0) + signal.at(1)) / 2, 2.12),
    text(size: 7pt, fill: ink, [signal]),
    anchor: "center",
  )
  content(
    ((ground-b.at(0) + ground-b.at(1)) / 2, 2.12),
    text(size: 7pt, fill: muted, [ground]),
    anchor: "center",
  )

  // Width is measured on the metal that remains, spacing on the etched slot.
  line(
    (signal.at(0), 1.25),
    (signal.at(1), 1.25),
    stroke: 0.8pt + dark-blue,
    mark: (start: ">", end: ">", size: 0.1, fill: dark-blue),
  )
  content(
    (signal.at(1) + 0.15, 1.25),
    text(
      size: 7.5pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [$w$],
    ),
    anchor: "west",
  )
  line(
    ((slot-a.at(0) + slot-a.at(1)) / 2, 3.45),
    ((slot-a.at(0) + slot-a.at(1)) / 2, 2.87),
    stroke: 0.8pt + dark-blue,
    mark: (end: ">", size: 0.09, fill: dark-blue),
  )
  content(
    ((slot-a.at(0) + slot-a.at(1)) / 2 - 0.15, 3.45),
    text(
      size: 7.5pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [$s$],
    ),
    anchor: "east",
  )
  content(
    (11.5, 0.6),
    text(size: 7pt, fill: muted, [mask model, rules, profiles, intent]),
    anchor: "center",
  )
})
