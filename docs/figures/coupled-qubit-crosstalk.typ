#import "@preview/cetz:0.4.2"
#import "style.typ": blue, body-font, dark-blue, heading-font, ink, muted, rule

#set page(width: 158mm, height: 64mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let copper = rgb("#a95a20")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.75),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [CROSSTALK THROUGH A PARASITIC COUPLING CAPACITANCE],
    ),
    anchor: "west",
  )
  content(
    (0.5, 5.32),
    text(
      size: 8pt,
      fill: muted,
      [A pulse on qubit 1 reaches qubit 2 only through $C_upright(m)$. The traces are a sketch.],
    ),
    anchor: "west",
  )

  let top = 3.75
  let bottom = 1.1

  let capacitor(x, color) = {
    line((x, top), (x, 2.55), stroke: 1.4pt + color)
    line((x - 0.38, 2.55), (x + 0.38, 2.55), stroke: 2pt + color)
    line((x - 0.38, 2.33), (x + 0.38, 2.33), stroke: 2pt + color)
    line((x, 2.33), (x, bottom), stroke: 1.4pt + color)
  }
  let junction(x, color) = {
    line((x, top), (x, 2.75), stroke: 1.4pt + color)
    line((x - 0.3, 2.75), (x + 0.3, 2.15), stroke: 1.6pt + color)
    line((x + 0.3, 2.75), (x - 0.3, 2.15), stroke: 1.6pt + color)
    line((x, 2.15), (x, bottom), stroke: 1.4pt + color)
  }

  // Pulse source and 50 Ω drive resistor.
  circle((1.0, 2.43), radius: 0.42, stroke: 1.3pt + ink)
  line(
    (0.75, 2.25),
    (0.88, 2.25),
    (0.88, 2.62),
    (1.12, 2.62),
    (1.12, 2.25),
    (1.25, 2.25),
    stroke: 1pt + ink,
  )
  line((1.0, 2.85), (1.0, top), (1.6, top), stroke: 1.4pt + ink)
  line((1.0, 2.01), (1.0, bottom), stroke: 1.4pt + ink)
  let zig = ((1.6, top),)
  for i in range(6) {
    zig.push((1.7 + i * 0.16, top + if calc.even(i) { 0.17 } else { -0.17 }))
  }
  zig.push((2.66, top))
  line(..zig, stroke: 1.3pt + ink)
  content(
    (2.13, top + 0.48),
    text(size: 8pt, fill: muted, [50 Ω]),
    anchor: "center",
  )
  line((2.66, top), (5.25, top), stroke: 1.4pt + ink)

  // Qubit 1, the coupling capacitor, and qubit 2.
  capacitor(3.4, blue)
  junction(4.6, blue)
  line((5.25, top), (5.55, top), stroke: 1.4pt + copper)
  line((5.55, top - 0.42), (5.55, top + 0.42), stroke: 2pt + copper)
  line((5.77, top - 0.42), (5.77, top + 0.42), stroke: 2pt + copper)
  line((5.77, top), (6.07, top), stroke: 1.4pt + copper)
  content(
    (5.66, top + 0.75),
    text(size: 10pt, fill: copper, $C_upright(m)$),
    anchor: "center",
  )
  line((6.07, top), (8.0, top), stroke: 1.4pt + ink)
  capacitor(6.8, dark-blue)
  junction(8.0, dark-blue)
  line((1.0, bottom), (8.0, bottom), stroke: 1.4pt + ink)
  for x in (3.4, 4.6, 6.8, 8.0) {
    circle((x, top), radius: 0.08, fill: ink, stroke: none)
  }
  content(
    (4.0, 0.62),
    text(size: 9pt, font: heading-font, weight: "bold", fill: blue, [qubit 1]),
    anchor: "center",
  )
  content(
    (7.4, 0.62),
    text(
      size: 9pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [qubit 2],
    ),
    anchor: "center",
  )

  // Sketch of the transient: drive on qubit 1, induced response on qubit 2.
  let x0 = 9.6
  let x1 = 15.0
  for (y, label, color) in (
    (3.35, [drive on qubit 1], blue),
    (1.25, [induced on qubit 2], dark-blue),
  ) {
    line((x0, y), (x1, y), stroke: 0.6pt + rule)
    content(
      (x0, y + 1.15),
      text(size: 8pt, fill: color, label),
      anchor: "west",
    )
  }
  line(
    (x0, 3.35),
    (10.6, 3.35),
    (10.75, 4.2),
    (12.6, 4.2),
    (12.75, 3.35),
    (x1, 3.35),
    stroke: 1.5pt + blue,
  )
  let response = ()
  for i in range(241) {
    let x = x0 + i * (x1 - x0) / 240
    let edge(t0) = if x < t0 { 0 } else {
      calc.exp(-(x - t0) / 0.9) * calc.sin(18 * (x - t0))
    }
    response.push((x, 1.25 + 0.42 * (edge(10.68) - edge(12.68))))
  }
  line(..response, stroke: 1.3pt + dark-blue)
  content(
    ((x0 + x1) / 2, 0.62),
    text(size: 8pt, fill: muted, [time]),
    anchor: "center",
  )
})
