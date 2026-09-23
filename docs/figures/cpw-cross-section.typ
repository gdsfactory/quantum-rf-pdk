#import "@preview/cetz:0.4.2"
#import "style.typ": blue, cyan, ink, muted, paper, purple, rule

#set page(width: 140mm, height: 63mm, margin: 3mm, fill: white)
#set text(font: "Arial", fill: ink)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.2),
    text(size: 8pt, weight: "bold", fill: blue, [CPW CROSS SECTION]),
    anchor: "west",
  )
  content(
    (0.5, 4.75),
    text(
      size: 8pt,
      fill: muted,
      [The center strip and the two slots set the line impedance.],
    ),
    anchor: "west",
  )

  rect((0.5, 0.7), (12.6, 2.95), fill: paper, stroke: 0.7pt + rule)
  content(
    (6.55, 1.65),
    text(size: 11pt, fill: muted, [substrate  εᵣ]),
    anchor: "center",
  )
  line((0.5, 2.95), (12.6, 2.95), stroke: 0.7pt + rule)

  rect((0.5, 2.95), (3.6, 3.3), fill: blue, stroke: none)
  rect((4.65, 2.95), (8.45, 3.3), fill: cyan, stroke: none)
  rect((9.5, 2.95), (12.6, 3.3), fill: blue, stroke: none)
  content((2.05, 3.6), text(size: 8pt, fill: blue, [ground]), anchor: "center")
  content((6.55, 3.6), text(size: 8pt, fill: cyan, [signal]), anchor: "center")
  content((11.05, 3.6), text(size: 8pt, fill: blue, [ground]), anchor: "center")

  line((4.65, 4.25), (8.45, 4.25), stroke: 0.8pt + ink, mark: (
    start: ">",
    end: ">",
    size: 0.13,
    fill: ink,
  ))
  content(
    (6.55, 4.48),
    text(size: 8pt, weight: "bold", [w: center width]),
    anchor: "center",
  )
  line((3.6, 2.48), (4.65, 2.48), stroke: 0.8pt + purple, mark: (
    start: ">",
    end: ">",
    size: 0.12,
    fill: purple,
  ))
  line((8.45, 2.48), (9.5, 2.48), stroke: 0.8pt + purple, mark: (
    start: ">",
    end: ">",
    size: 0.12,
    fill: purple,
  ))
  content(
    (4.12, 2.18),
    text(size: 8pt, weight: "bold", fill: purple, [s]),
    anchor: "center",
  )
  content(
    (8.98, 2.18),
    text(size: 8pt, weight: "bold", fill: purple, [s]),
    anchor: "center",
  )
})
