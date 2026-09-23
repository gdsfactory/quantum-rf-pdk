#import "@preview/cetz:0.4.2"
#import "style.typ": blue, coral, cyan, ink, muted, paper, purple, rule

#set page(width: 180mm, height: 56mm, margin: 3mm, fill: white)
#set text(font: "Arial", fill: ink)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let card(x, y, number, title, detail, color) = {
    rect(
      (x, y),
      (x + 3.65, y + 1.75),
      radius: 0.17,
      fill: paper,
      stroke: 0.8pt + rule,
    )
    rect((x, y + 1.61), (x + 3.65, y + 1.75), fill: color, stroke: none)
    content(
      (x + 0.38, y + 1.29),
      text(size: 8pt, weight: "bold", fill: color, number),
      anchor: "west",
    )
    content(
      (x + 0.38, y + 0.92),
      text(size: 10pt, weight: "bold", title),
      anchor: "west",
    )
    content(
      (x + 0.38, y + 0.44),
      text(size: 7.5pt, fill: muted, detail),
      anchor: "west",
    )
  }

  let arrow(a, b) = line(a, b, stroke: 1.2pt + blue, mark: (
    end: ">",
    size: 0.16,
    fill: blue,
  ))

  card(0.35, 3.65, "01", [Requirements], [frequency · coupling · T1], blue)
  card(4.55, 3.65, "02", [Hamiltonian], [qubit · resonator values], purple)
  card(8.75, 3.65, "03", [SAX circuit], [microwave response], cyan)
  card(12.95, 3.65, "04", [Layout], [geometry · routing], coral)
  arrow((4.05, 4.53), (4.45, 4.53))
  arrow((8.25, 4.53), (8.65, 4.53))
  arrow((12.45, 4.53), (12.85, 4.53))

  card(12.95, 0.65, "05", [EM verification], [fields · modes], blue)
  card(8.75, 0.65, "06", [Pulse simulation], [gates · leakage · noise], purple)
  card(4.55, 0.65, "07", [Measurement], [compare with targets], cyan)
  arrow((14.78, 3.61), (14.78, 2.46))
  arrow((12.9, 1.53), (12.5, 1.53))
  arrow((8.7, 1.53), (8.3, 1.53))
})
