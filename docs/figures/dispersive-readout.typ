#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 164mm, height: 58mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.2),
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
    (0.5, 4.72),
    text(
      size: 8pt,
      fill: muted,
      [The qubit state shifts the resonance seen by a weak readout tone.],
    ),
    anchor: "west",
  )

  rect(
    (0.55, 1.05),
    (5.15, 4.15),
    radius: 0.15,
    fill: paper,
    stroke: 0.8pt + rule,
  )
  content(
    (2.85, 3.66),
    text(size: 10pt, font: heading-font, weight: "bold", [qubit]),
    anchor: "center",
  )
  content(
    (2.85, 2.88),
    text(size: 16pt, fill: dark-blue, [$|0⟩$  or  $|1⟩$]),
    anchor: "center",
  )
  content(
    (2.85, 1.8),
    text(size: 8pt, fill: muted, [coupled to the readout resonator]),
    anchor: "center",
  )
  line((5.3, 2.6), (6.4, 2.6), stroke: 1.3pt + blue, mark: (
    end: ">",
    size: 0.17,
    fill: blue,
  ))

  line((7.0, 1.32), (14.85, 1.32), stroke: 0.9pt + ink, mark: (
    end: ">",
    size: 0.15,
    fill: ink,
  ))
  line((7.0, 1.32), (7.0, 4.05), stroke: 0.9pt + ink)
  content(
    (11.0, 0.88),
    text(size: 8pt, fill: muted, [probe frequency]),
    anchor: "center",
  )
  content(
    (7.25, 3.85),
    text(size: 8pt, fill: muted, [transmission]),
    anchor: "west",
  )

  let left = ()
  let right = ()
  for i in range(81) {
    let x = 7.3 + i * 0.09
    left.push((
      x,
      3.52 - 1.5 / (1 + ((x - 10.25) / 0.34) * ((x - 10.25) / 0.34)),
    ))
    right.push((
      x,
      3.52 - 1.5 / (1 + ((x - 11.65) / 0.34) * ((x - 11.65) / 0.34)),
    ))
  }
  line(..left, stroke: 1.5pt + light-blue)
  line(..right, stroke: 1.5pt + dark-blue)
  line((10.25, 1.35), (10.25, 1.93), stroke: 0.7pt + light-blue)
  line((11.65, 1.35), (11.65, 1.93), stroke: 0.7pt + dark-blue)
  content(
    (9.75, 4.16),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: light-blue,
      [$|0⟩$ response],
    ),
    anchor: "center",
  )
  content(
    (12.3, 4.16),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: dark-blue,
      [$|1⟩$ response],
    ),
    anchor: "center",
  )
})
