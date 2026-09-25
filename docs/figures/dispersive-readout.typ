#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, muted, paper, rule,
)

#set page(width: 164mm, height: 95mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let copper = rgb("#a95a20")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 8.7),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [DISPERSIVE READOUT],
    ),
    anchor: "west",
  )
  content(
    (0.5, 8.22),
    text(
      size: 8pt,
      fill: muted,
      [The qubit state shifts the resonance seen by a weak readout tone.],
    ),
    anchor: "west",
  )

  rect(
    (0.55, 1.8),
    (5.15, 7.5),
    radius: 0.15,
    fill: paper,
    stroke: 0.8pt + rule,
  )
  content(
    (2.85, 6.45),
    text(size: 10pt, font: heading-font, weight: "bold", [qubit]),
    anchor: "center",
  )
  content(
    (2.85, 4.7),
    text(size: 16pt, fill: dark-blue, [$|0⟩$  or  $|1⟩$]),
    anchor: "center",
  )
  content(
    (2.85, 2.85),
    text(size: 8pt, fill: muted, [coupled to the readout resonator]),
    anchor: "center",
  )
  line((5.3, 4.65), (6.4, 4.65), stroke: 1.3pt + blue, mark: (
    end: ">",
    size: 0.17,
    fill: blue,
  ))

  line((7.0, 1.8), (14.85, 1.8), stroke: 0.9pt + ink, mark: (
    end: ">",
    size: 0.15,
    fill: ink,
  ))
  line((7.0, 1.8), (7.0, 7.3), stroke: 0.9pt + ink)
  content(
    (11.0, 1.25),
    text(size: 8pt, fill: muted, [probe frequency]),
    anchor: "center",
  )
  content(
    (7.25, 7.0),
    text(size: 8pt, fill: muted, [transmission]),
    anchor: "west",
  )

  let left = ()
  let right = ()
  for i in range(241) {
    let x = 7.3 + i * 0.03
    left.push((
      x,
      6.55 - 3.0 / (1 + ((x - 10.25) / 0.34) * ((x - 10.25) / 0.34)),
    ))
    right.push((
      x,
      6.55 - 3.0 / (1 + ((x - 11.65) / 0.34) * ((x - 11.65) / 0.34)),
    ))
  }
  line(..left, stroke: 1.5pt + blue)
  line(..right, stroke: 1.5pt + copper)
  line((10.25, 1.82), (10.25, 3.45), stroke: 0.7pt + blue)
  line((11.65, 1.82), (11.65, 3.45), stroke: 0.7pt + copper)
  content(
    (9.65, 7.6),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [$|0⟩$ response],
    ),
    anchor: "center",
  )
  content(
    (12.3, 7.6),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: copper,
      [$|1⟩$ response],
    ),
    anchor: "center",
  )
})
