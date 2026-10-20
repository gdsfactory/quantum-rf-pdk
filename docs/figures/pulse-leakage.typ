#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 161mm, height: 52mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.03),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [PULSE CONTROL AND LEAKAGE],
    ),
    anchor: "west",
  )
  content(
    (0.5, 4.59),
    text(
      size: 8pt,
      fill: muted,
      [A finite pulse drives the qubit transition and can populate the next level.],
    ),
    anchor: "west",
  )

  content(
    (3.7, 3.8),
    text(
      size: 10pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [microwave pulse],
    ),
    anchor: "center",
  )
  line((0.9, 1.35), (7.2, 1.35), stroke: 0.8pt + ink, mark: (
    end: ">",
    size: 0.14,
    fill: ink,
  ))
  line((0.9, 1.35), (0.9, 3.45), stroke: 0.8pt + ink)
  let pulse = ()
  for i in range(61) {
    let x = 1.05 + i * 0.096
    let u = (x - 3.6) / 1.15
    pulse.push((x, 1.4 + 2.05 * calc.exp(-u * u)))
  }
  line(..pulse, stroke: 2pt + light-blue)
  content((6.8, 0.95), text(size: 8pt, fill: muted, [time]), anchor: "east")

  line((7.5, 2.5), (8.3, 2.5), stroke: 1.1pt + blue, mark: (
    end: ">",
    size: 0.16,
    fill: blue,
  ))

  content(
    (11.5, 3.86),
    text(
      size: 10pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [transmon levels],
    ),
    anchor: "center",
  )
  line((9.3, 1.25), (13.2, 1.25), stroke: 1.6pt + ink)
  line((9.3, 2.53), (13.2, 2.53), stroke: 1.6pt + ink)
  line((9.3, 3.47), (13.2, 3.47), stroke: 1.6pt + ink)
  content(
    (13.55, 1.25),
    text(size: 9pt, font: heading-font, weight: "bold", [$|0⟩$]),
    anchor: "west",
  )
  content(
    (13.55, 2.53),
    text(size: 9pt, font: heading-font, weight: "bold", [$|1⟩$]),
    anchor: "west",
  )
  content(
    (13.55, 3.47),
    text(size: 9pt, font: heading-font, weight: "bold", [$|2⟩$]),
    anchor: "west",
  )
  line((10.1, 1.4), (10.1, 2.37), stroke: 1.6pt + blue, mark: (
    end: ">",
    size: 0.16,
    fill: blue,
  ))
  line((12.25, 2.67), (12.25, 3.29), stroke: 1.6pt + slate, mark: (
    end: ">",
    size: 0.16,
    fill: slate,
  ))
  content((10.1, 0.72), text(size: 8pt, fill: blue, [gate]), anchor: "center")
  content(
    (12.25, 0.72),
    text(size: 8pt, fill: slate, [leakage]),
    anchor: "center",
  )
})
