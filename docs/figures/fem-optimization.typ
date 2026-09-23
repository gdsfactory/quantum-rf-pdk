#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 171mm, height: 50mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let card(x, title, detail, color) = {
    rect(
      (x, 2.0),
      (x + 3.55, 4.05),
      radius: 0.15,
      fill: paper,
      stroke: 0.8pt + rule,
    )
    rect((x, 3.92), (x + 3.55, 4.05), fill: color, stroke: none)
    content(
      (x + 0.26, 3.35),
      text(size: 9pt, font: heading-font, weight: "bold", fill: color, title),
      anchor: "west",
    )
    content(
      (x + 0.26, 2.68),
      text(size: 7.5pt, fill: muted, detail),
      anchor: "west",
    )
  }
  let arrow(a, b) = line(a, b, stroke: 1.1pt + blue, mark: (
    end: ">",
    size: 0.15,
    fill: blue,
  ))

  content(
    (0.5, 4.72),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [OPTUNA + PALACE CAPACITOR DESIGN LOOP],
    ),
    anchor: "west",
  )
  card(0.5, [Suggest geometry], [length · gap · thickness], blue)
  card(4.65, [Build & mesh], [five-finger capacitor], light-blue)
  card(8.8, [Solve fields], [Palace capacitance], dark-blue)
  card(12.95, [Score trial], [distance from 40 fF], slate)
  arrow((4.1, 3.0), (4.55, 3.0))
  arrow((8.25, 3.0), (8.7, 3.0))
  arrow((12.4, 3.0), (12.85, 3.0))

  line((14.72, 1.9), (14.72, 1.07), stroke: 1.1pt + blue)
  line((14.72, 1.07), (2.27, 1.07), stroke: 1.1pt + blue)
  arrow((2.27, 1.07), (2.27, 1.9))
  rect((6.3, 0.77), (10.75, 1.37), radius: 0.12, fill: white, stroke: none)
  content(
    (8.5, 1.08),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [next Optuna trial],
    ),
    anchor: "center",
  )
})
