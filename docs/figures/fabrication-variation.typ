#import "@preview/cetz:0.4.2"
#import "style.typ": ink, muted, blue, cyan, purple, coral, paper, rule

#set page(width: 165mm, height: 62mm, margin: 3mm, fill: white)
#set text(font: "Arial", fill: ink)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content((0.5, 5.7), text(size: 8pt, weight: "bold", fill: blue, [TWO WAYS FABRICATION SHIFTS A RESONATOR ARRAY]), anchor: "west")
  content((0.5, 5.22), text(size: 8pt, fill: muted, [Each tick is a resonant frequency; the arrows show changes from one fabrication draw.]), anchor: "west")

  let axis(y, title, color) = {
    content((0.55, y + 0.44), text(size: 9pt, weight: "bold", fill: color, title), anchor: "west")
    line((4.25, y), (14.9, y), stroke: 0.8pt + rule, mark: (end: ">", size: 0.13, fill: rule))
  }
  let tick(x, y, color) = line((x, y - 0.22), (x, y + 0.22), stroke: 2pt + color)
  let nominal = (5.4, 7.45, 9.5, 11.55, 13.6)

  axis(4.25, [Nominal], muted)
  for x in nominal { tick(x, 4.25, muted) }

  axis(2.95, [Global variation], blue)
  for x in nominal {
    line((x, 4.0), (x + 0.52, 3.19), stroke: 0.6pt + blue.transparentize(55%))
    tick(x + 0.52, 2.95, blue)
  }
  content((14.75, 2.6), text(size: 7pt, fill: blue, [shared shift]), anchor: "east")

  axis(1.45, [Local variation], purple)
  let offsets = (-0.35, 0.45, -0.15, 0.3, -0.4)
  for (i, x) in nominal.enumerate() {
    let shifted = x + offsets.at(i)
    line((x, 4.0), (shifted, 1.69), stroke: 0.6pt + purple.transparentize(65%))
    tick(shifted, 1.45, purple)
  }
  content((14.75, 1.1), text(size: 7pt, fill: purple, [independent shifts]), anchor: "east")
  content((14.8, 0.58), text(size: 8pt, fill: muted, [frequency]), anchor: "east")
})
