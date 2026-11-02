#import "@preview/cetz:0.4.2"
#import "style.typ": blue, body-font, heading-font, ink, muted, rule

#set page(width: 158mm, height: 64mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#let magenta = rgb("#c026d3")
#let copper = rgb("#a95a20")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 5.9),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [DECLARING INTENDED COUPLING],
    ),
    anchor: "west",
  )
  content(
    (0.5, 5.5),
    text(
      size: 8pt,
      fill: muted,
      [A marker tells the deck to expect the gap. Without one, the pair is crosstalk.],
    ),
    anchor: "west",
  )

  let draw-panel(origin, marker) = {
    let (left, bottom) = origin
    let right = left + 7.1
    let top = bottom + 4.1

    rect((left, bottom), (right, top), stroke: 0.6pt + rule)
    line((left + 0.7, 3.75), (right - 0.7, 3.75), stroke: 5pt + blue)
    line((left + 0.7, 2.55), (right - 0.7, 2.55), stroke: 5pt + blue)
    if marker {
      rect(
        (left + 0.7, 2.64),
        (right - 0.7, 3.66),
        fill: magenta.transparentize(88%),
        stroke: 0.7pt + magenta,
      )
      content(
        ((left + right) / 2, 3.15),
        text(size: 7.5pt, fill: magenta, [COUPLING 103/0]),
        anchor: "center",
      )
    } else {
      rect(
        (left + 0.7, 2.64),
        (right - 0.7, 3.66),
        stroke: (paint: copper, thickness: 0.8pt, dash: "dashed"),
      )
      circle(
        ((left + right) / 2, 4.35),
        radius: 0.18,
        fill: copper,
        stroke: none,
      )
      content(
        ((left + right) / 2, 4.35),
        text(size: 7.5pt, font: heading-font, weight: "bold", fill: white, [!]),
        anchor: "center",
      )
    }
    // Edge-to-edge gap.
    line(
      (right - 0.35, 2.64),
      (right - 0.35, 3.66),
      stroke: 0.7pt + muted,
      mark: (start: ">", end: ">", size: 0.1, fill: muted),
    )
    content(
      (right - 0.15, 3.15),
      text(size: 7.5pt, font: heading-font, weight: "bold", fill: muted, [$g$]),
      anchor: "west",
    )
  }

  draw-panel((0.5, 0.8), true)
  content(
    (4.05, 1.3),
    text(size: 8pt, font: heading-font, weight: "bold", fill: blue, [declared]),
    anchor: "center",
  )
  content(
    (4.05, 1.0),
    text(size: 7.5pt, fill: muted, [RF.C.1 waives the gap]),
    anchor: "center",
  )

  draw-panel((8.2, 0.8), false)
  content(
    (11.75, 1.3),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: copper,
      [undeclared],
    ),
    anchor: "center",
  )
  content(
    (11.75, 1.0),
    text(size: 7.5pt, fill: muted, [RF.C.1 reports the pair]),
    anchor: "center",
  )
})
