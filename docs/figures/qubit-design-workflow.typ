#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, muted, paper, rule,
)

#set page(width: 158mm, height: 46mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  content(
    (0.5, 4.1),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [FROM A READOUT TARGET TO LAYOUT PARAMETERS],
    ),
    anchor: "west",
  )
  content(
    (0.5, 3.67),
    text(
      size: 8pt,
      fill: muted,
      [Each stage fixes the quantities that the next stage needs.],
    ),
    anchor: "west",
  )

  let stages = (
    (
      [steps 1–2],
      [targets],
      [$χ$, $α$],
      [with qubit and resonator frequencies],
    ),
    ([step 3], [coupling], [$g$], [solved from the target $χ$]),
    (
      [step 4],
      [circuit],
      [$C_Σ$, $L_upright(J)$, $C_upright(c)$],
      [from qpdk conversion helpers],
    ),
    ([step 5], [layout], [length], [of the readout resonator]),
  )
  let width = 3.05
  let pitch = 3.75
  for (i, (step, title, body, note)) in stages.enumerate() {
    let x = 0.55 + i * pitch
    let last = i == stages.len() - 1
    let accent = if last { dark-blue } else { blue }
    content(
      (x + width / 2, 2.95),
      text(size: 7pt, fill: muted, step),
      anchor: "center",
    )
    rect(
      (x, 0.45),
      (x + width, 2.6),
      radius: 0.12,
      fill: paper,
      stroke: 0.8pt + if last { dark-blue } else { rule },
    )
    content(
      (x + width / 2, 2.05),
      text(
        size: 8pt,
        font: heading-font,
        weight: "bold",
        fill: accent,
        upper(title),
      ),
      anchor: "center",
    )
    content(
      (x + width / 2, 1.38),
      text(size: 10pt, body),
      anchor: "center",
    )
    content(
      (x + width / 2, 0.82),
      box(width: (width - 0.3) * 1cm, align(center, text(
        size: 6.5pt,
        fill: muted,
        note,
      ))),
      anchor: "center",
    )
    if not last {
      line(
        (x + width + 0.12, 1.52),
        (x + pitch - 0.12, 1.52),
        stroke: 1.1pt + blue,
        mark: (end: ">", size: 0.15, fill: blue),
      )
    }
  }
})
