#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
)

#set page(width: 140mm, height: 80mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 6.9),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [CPW CROSS SECTION],
    ),
    anchor: "west",
  )
  content(
    (0.5, 6.45),
    text(
      size: 8pt,
      fill: muted,
      [The center strip and the two slots set the line impedance.],
    ),
    anchor: "west",
  )

  rect((0.5, 1.25), (12.6, 3.55), fill: paper, stroke: 0.7pt + rule)
  content(
    (6.55, 2.0),
    text(size: 11pt, fill: muted, [substrate  εᵣ]),
    anchor: "center",
  )
  line((0.5, 3.55), (12.6, 3.55), stroke: 0.7pt + rule)

  rect((0.5, 3.55), (3.6, 4.1), fill: blue, stroke: none)
  rect((4.65, 3.55), (8.45, 4.1), fill: light-blue, stroke: none)
  rect((9.5, 3.55), (12.6, 4.1), fill: blue, stroke: none)
  content((2.05, 4.45), text(size: 8pt, fill: blue, [ground]), anchor: "center")
  content(
    (6.55, 4.45),
    text(size: 8pt, fill: light-blue, [signal]),
    anchor: "center",
  )
  content(
    (11.05, 4.45),
    text(size: 8pt, fill: blue, [ground]),
    anchor: "center",
  )

  line((4.65, 5.15), (8.45, 5.15), stroke: 0.8pt + ink, mark: (
    start: ">",
    end: ">",
    size: 0.13,
    fill: ink,
  ))
  content(
    (6.55, 5.45),
    text(size: 8pt, font: heading-font, weight: "bold", [w: center width]),
    anchor: "center",
  )
  line((3.6, 2.98), (4.65, 2.98), stroke: 0.8pt + dark-blue, mark: (
    start: ">",
    end: ">",
    size: 0.12,
    fill: dark-blue,
  ))
  line((8.45, 2.98), (9.5, 2.98), stroke: 0.8pt + dark-blue, mark: (
    start: ">",
    end: ">",
    size: 0.12,
    fill: dark-blue,
  ))
  content(
    (4.12, 2.68),
    text(size: 8pt, font: heading-font, weight: "bold", fill: dark-blue, [s]),
    anchor: "center",
  )
  content(
    (8.98, 2.68),
    text(size: 8pt, font: heading-font, weight: "bold", fill: dark-blue, [s]),
    anchor: "center",
  )
  line((9.05, 3.55), (9.05, 4.1), stroke: 0.8pt + ink, mark: (
    start: ">",
    end: ">",
    size: 0.1,
    fill: ink,
  ))
  content(
    (9.27, 3.83),
    text(size: 8pt, font: heading-font, weight: "bold", [$t$]),
    anchor: "west",
  )
  content(
    (10.95, 5.45),
    text(size: 8pt, fill: blue, [superconductor]),
    anchor: "center",
  )
  line((10.95, 5.2), (11.9, 4.1), stroke: 0.7pt + blue)
})
