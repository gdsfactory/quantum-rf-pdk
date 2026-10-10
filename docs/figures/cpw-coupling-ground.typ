#import "@preview/cetz:0.4.2"
#import "style.typ": blue, body-font, dark-blue, heading-font, ink, muted, paper

#set page(width: 160mm, height: 83mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  for (offset, grounded) in ((0, false), (7.6, true)) {
    let x = offset + 0.2
    content((x + 3.1, 7.0), text(
      font: heading-font,
      size: 10pt,
      weight: "bold",
      if grounded { [Drawn CPW slots] } else { [Fully etched inner gap] },
    ))
    rect((x, 1.0), (x + 5.9, 6.3), fill: paper, stroke: none)
    for (low, high) in ((1.0, 1.8), (5.5, 6.3)) {
      rect((x, low), (x + 5.9, high), fill: dark-blue, stroke: none)
    }
    for (low, high) in ((2.4, 2.9), (4.4, 4.9)) {
      rect((x, low), (x + 5.9, high), fill: blue, stroke: none)
    }
    if grounded {
      rect((x, 3.5), (x + 5.9, 3.8), fill: dark-blue, stroke: none)
      content((x + 2.95, 3.65), text(
        size: 7pt,
        fill: white,
        [ground strip: $g - 2s$],
      ))
      content((x + 2.95, 3.2), text(size: 7pt, fill: muted, [etched slot]))
    } else {
      content((x + 2.95, 3.65), text(
        size: 8pt,
        fill: muted,
        [etched inner gap],
      ))
    }
    content((x + 2.95, 2.65), text(size: 8pt, fill: white, [lower trace]))
    content((x + 2.95, 4.65), text(size: 8pt, fill: white, [upper trace]))
    content((x + 2.95, 5.9), text(size: 8pt, fill: white, [ground]))
    for (low, high, label) in (
      (2.9, 4.4, [$g$]),
      (2.4, 2.9, [$w$]),
      (1.8, 2.4, [$s$]),
    ) {
      line((x + 6.15, low), (x + 6.15, high), stroke: 0.6pt + muted)
      line((x + 6.03, low), (x + 6.27, low), stroke: 0.6pt + muted)
      line((x + 6.03, high), (x + 6.27, high), stroke: 0.6pt + muted)
      content((x + 6.55, (low + high) / 2), text(size: 8pt, label))
    }
  }
  content((7.4, 0.4), text(
    size: 8pt,
    fill: muted,
    [Top view, schematic. Same trace width $w$, slot width $s$ and gap $g > 2s$.],
  ))
})
