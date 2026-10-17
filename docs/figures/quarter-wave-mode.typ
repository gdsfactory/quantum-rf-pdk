#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, rule,
)

#set page(width: 158mm, height: 76mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let copper = rgb("#a95a20")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 6.55),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [FUNDAMENTAL MODE OF A QUARTER-WAVE RESONATOR],
    ),
    anchor: "west",
  )
  content(
    (0.5, 6.12),
    text(
      size: 8pt,
      fill: muted,
      [The open end holds a voltage maximum and the shorted end a current maximum.],
    ),
    anchor: "west",
  )

  let x0 = 2.0
  let x1 = 10.0
  let base = 2.0
  let amp = 2.85

  // Standing-wave envelopes along the line.
  line((x0, base), (x1, base), stroke: 0.6pt + rule)
  let voltage = ()
  let current = ()
  for i in range(201) {
    let u = i / 200
    let x = x0 + u * (x1 - x0)
    voltage.push((x, base + amp * calc.cos(u * calc.pi / 2)))
    current.push((x, base + amp * calc.sin(u * calc.pi / 2)))
  }
  line(..voltage, stroke: 1.6pt + blue)
  line(..current, stroke: 1.6pt + copper)
  content(
    (x0 + 0.25, base + amp + 0.25),
    text(size: 9pt, font: heading-font, weight: "bold", fill: blue, [$|V|$]),
    anchor: "west",
  )
  content(
    (x1 - 0.25, base + amp + 0.25),
    text(
      size: 9pt,
      font: heading-font,
      weight: "bold",
      fill: copper,
      [$|I|$],
    ),
    anchor: "east",
  )

  // The CPW centre conductor, open on the left and shorted on the right.
  line((x0, 1.1), (x1, 1.1), stroke: 2.2pt + dark-blue)
  circle((x0, 1.1), radius: 0.13, fill: white, stroke: 1.5pt + dark-blue)
  line((x1, 1.1), (x1, 0.65), stroke: 1.5pt + ink)
  line((x1 - 0.35, 0.65), (x1 + 0.35, 0.65), stroke: 1.2pt + ink)
  line((x1 - 0.23, 0.48), (x1 + 0.23, 0.48), stroke: 1.2pt + ink)
  line((x1 - 0.11, 0.31), (x1 + 0.11, 0.31), stroke: 1.2pt + ink)
  content(
    (x0 - 0.3, 1.1),
    text(size: 9pt, font: heading-font, weight: "bold", [open]),
    anchor: "east",
  )
  content(
    (x1 + 0.5, 0.65),
    text(size: 9pt, font: heading-font, weight: "bold", [short]),
    anchor: "west",
  )
  line((x0, -0.2), (x1, -0.2), stroke: 0.6pt + muted, mark: (
    start: ">",
    end: ">",
    size: 0.13,
    fill: muted,
  ))
  content(
    ((x0 + x1) / 2, -0.2),
    box(fill: white, inset: (x: 3pt), text(
      size: 9pt,
      [$L = λ slash 4$],
    )),
    anchor: "center",
  )

  rect(
    (11.0, 1.6),
    (15.0, 5.0),
    radius: 0.12,
    fill: light-blue.transparentize(86%),
    stroke: 0.8pt + light-blue,
  )
  content(
    (13.0, 4.55),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: light-blue,
      [RESONANCE],
    ),
    anchor: "center",
  )
  content(
    (13.0, 3.6),
    text(size: 10pt, $f = display(v_upright(p) / (4 L))$),
    anchor: "center",
  )
  content(
    (13.0, 2.4),
    text(size: 10pt, $v_upright(p) = display(c_0 / sqrt(ε_upright("eff")))$),
    anchor: "center",
  )
})
