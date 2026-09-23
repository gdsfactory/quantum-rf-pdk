#import "@preview/cetz:0.4.2"
#import "style.typ": blue, coral, cyan, ink, muted, paper, purple, rule

#set page(width: 158mm, height: 50mm, margin: 3mm, fill: white)
#set text(font: "Arial", fill: ink)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 4.9),
    text(
      size: 8pt,
      weight: "bold",
      fill: blue,
      [THE TWO ENERGY SCALES OF A TRANSMON],
    ),
    anchor: "west",
  )
  content(
    (0.5, 4.47),
    text(
      size: 8pt,
      fill: muted,
      [A Josephson junction shunted by capacitance forms a weakly anharmonic circuit.],
    ),
    anchor: "west",
  )

  line((1.3, 3.72), (6.65, 3.72), stroke: 1.8pt + ink)
  line((1.3, 1.05), (6.65, 1.05), stroke: 1.8pt + ink)
  circle((1.3, 3.72), radius: 0.11, fill: ink, stroke: none)
  circle((1.3, 1.05), radius: 0.11, fill: ink, stroke: none)
  line((3.0, 3.72), (3.0, 2.55), stroke: 1.5pt + cyan)
  line((2.47, 2.55), (3.53, 2.55), stroke: 2.2pt + cyan)
  line((2.47, 2.31), (3.53, 2.31), stroke: 2.2pt + cyan)
  line((3.0, 2.31), (3.0, 1.05), stroke: 1.5pt + cyan)
  content(
    (3.0, 0.62),
    text(size: 10pt, weight: "bold", fill: cyan, $C_Σ$),
    anchor: "center",
  )

  line((5.0, 3.72), (5.0, 2.7), stroke: 1.5pt + purple)
  line((4.57, 2.7), (5.43, 1.99), stroke: 1.7pt + purple)
  line((5.43, 2.7), (4.57, 1.99), stroke: 1.7pt + purple)
  line((5.0, 1.99), (5.0, 1.05), stroke: 1.5pt + purple)
  content(
    (5.0, 0.62),
    text(size: 10pt, weight: "bold", fill: purple, [junction]),
    anchor: "center",
  )

  line((6.9, 2.4), (7.85, 2.4), stroke: 1.1pt + blue, mark: (
    end: ">",
    size: 0.16,
    fill: blue,
  ))

  rect(
    (8.15, 2.65),
    (14.9, 3.83),
    radius: 0.12,
    fill: cyan.transparentize(86%),
    stroke: 0.8pt + cyan,
  )
  content(
    (8.47, 3.24),
    text(
      size: 10pt,
      weight: "bold",
      fill: cyan,
      [$C_Σ$ sets $E_C = e^2 / (2 C_Σ)$],
    ),
    anchor: "west",
  )
  rect(
    (8.15, 1.05),
    (14.9, 2.23),
    radius: 0.12,
    fill: purple.transparentize(86%),
    stroke: 0.8pt + purple,
  )
  content(
    (8.47, 1.64),
    text(
      size: 10pt,
      weight: "bold",
      fill: purple,
      [$I_c$ sets $E_J = Φ_0 I_c / (2 π)$],
    ),
    anchor: "west",
  )
})
