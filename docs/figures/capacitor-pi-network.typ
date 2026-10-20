#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted,
)

#set page(width: 158mm, height: 58mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.75),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [A TWO-TERMINAL CAPACITOR AS A PI NETWORK],
    ),
    anchor: "west",
  )
  content(
    (0.5, 5.32),
    text(
      size: 8pt,
      fill: muted,
      [The series branch is the coupling; each terminal also sees the grounded metal around it.],
    ),
    anchor: "west",
  )

  // Series branch between the two terminals.
  line((0.9, 3.9), (3.55, 3.9), stroke: 1.6pt + ink)
  line((3.55, 3.9), (3.9, 3.9), stroke: 1.5pt + blue)
  line((3.9, 3.45), (3.9, 4.35), stroke: 2.2pt + blue)
  line((4.14, 3.45), (4.14, 4.35), stroke: 2.2pt + blue)
  line((4.14, 3.9), (4.49, 3.9), stroke: 1.5pt + blue)
  line((4.49, 3.9), (7.15, 3.9), stroke: 1.6pt + ink)
  content(
    (4.02, 4.72),
    text(size: 10pt, fill: blue, $C_12^upright("mutual")$),
    anchor: "center",
  )
  for x in (1.9, 6.15) {
    circle((x, 3.9), radius: 0.1, fill: ink, stroke: none)
  }
  content(
    (0.9, 4.25),
    text(size: 9pt, font: heading-font, weight: "bold", [port 1]),
    anchor: "west",
  )
  content(
    (7.15, 4.25),
    text(size: 9pt, font: heading-font, weight: "bold", [port 2]),
    anchor: "east",
  )

  // Shunt branches to ground.
  for (x, label) in (
    (1.9, $C_(1 upright(g))$),
    (6.15, $C_(2 upright(g))$),
  ) {
    line((x, 3.9), (x, 2.68), stroke: 1.5pt + dark-blue)
    line((x - 0.45, 2.68), (x + 0.45, 2.68), stroke: 2.2pt + dark-blue)
    line((x - 0.45, 2.44), (x + 0.45, 2.44), stroke: 2.2pt + dark-blue)
    line((x, 2.44), (x, 1.35), stroke: 1.5pt + dark-blue)
    content(
      (x + 0.62, 2.56),
      text(size: 10pt, fill: dark-blue, label),
      anchor: "west",
    )
  }
  line((1.9, 1.35), (6.15, 1.35), stroke: 1.6pt + ink)
  line((3.67, 1.35), (3.67, 1.1), stroke: 1.2pt + ink)
  line((3.32, 1.1), (4.02, 1.1), stroke: 1.2pt + ink)
  line((3.44, 0.93), (3.9, 0.93), stroke: 1.2pt + ink)
  line((3.56, 0.76), (3.78, 0.76), stroke: 1.2pt + ink)
  content(
    (4.25, 1.0),
    text(size: 8pt, fill: muted, [ground]),
    anchor: "west",
  )

  for y in (3.95, 1.75) {
    line((7.55, 2.85), (8.45, y), stroke: 1.1pt + blue, mark: (
      end: ">",
      size: 0.16,
      fill: blue,
    ))
  }

  // What each solver reports.
  rect(
    (8.6, 3.05),
    (15.0, 4.85),
    radius: 0.12,
    fill: dark-blue.transparentize(86%),
    stroke: 0.8pt + dark-blue,
  )
  content(
    (8.9, 4.48),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [MAXWELL CAPACITANCE MATRIX],
    ),
    anchor: "west",
  )
  content(
    (8.9, 3.68),
    text(
      size: 9pt,
      [$C_11 = C_(1 upright(g)) + C_12^upright("mutual")$, #h(0.6em) $C_12 = -C_12^upright("mutual")$],
    ),
    anchor: "west",
  )
  rect(
    (8.6, 0.85),
    (15.0, 2.65),
    radius: 0.12,
    fill: light-blue.transparentize(86%),
    stroke: 0.8pt + light-blue,
  )
  content(
    (8.9, 2.28),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: light-blue,
      [TWO-PORT ADMITTANCE],
    ),
    anchor: "west",
  )
  content(
    (8.9, 1.48),
    text(
      size: 9pt,
      [$Y_12 = -j ω C_12^upright("mutual")$, #h(0.6em) so $C_12^upright("mutual") = -op("Im") Y_12 slash ω$],
    ),
    anchor: "west",
  )
})
