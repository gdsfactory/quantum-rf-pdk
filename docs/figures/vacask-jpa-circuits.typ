#import "@preview/cetz:0.4.2"
#import "style.typ": (
  blue, body-font, dark-blue, heading-font, ink, light-blue, muted, paper, rule,
  slate,
)

#set page(width: 158mm, height: 62mm, margin: 3mm, fill: white)
#set text(font: body-font, fill: ink)
#show math.equation: set text(font: "Fira Math")

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let label(pos, body, fill: ink, size: 9pt, anchor: "center") = content(
    pos,
    text(size: size, font: heading-font, weight: "bold", fill: fill, body),
    anchor: anchor,
  )
  let note(pos, body, anchor: "center") = content(
    pos,
    text(size: 8pt, fill: muted, body),
    anchor: anchor,
  )
  // Vertical capacitor between (x, y-top) and (x, y-bottom)
  let vcap(x, top, bottom, color) = {
    let mid = (top + bottom) / 2
    line((x, top), (x, mid + 0.12), stroke: 1.5pt + color)
    line((x - 0.42, mid + 0.12), (x + 0.42, mid + 0.12), stroke: 2pt + color)
    line((x - 0.42, mid - 0.12), (x + 0.42, mid - 0.12), stroke: 2pt + color)
    line((x, mid - 0.12), (x, bottom), stroke: 1.5pt + color)
  }
  // Horizontal capacitor between (left, y) and (right, y)
  let hcap(left, right, y, color) = {
    let mid = (left + right) / 2
    line((left, y), (mid - 0.12, y), stroke: 1.5pt + color)
    line((mid - 0.12, y - 0.38), (mid - 0.12, y + 0.38), stroke: 2pt + color)
    line((mid + 0.12, y - 0.38), (mid + 0.12, y + 0.38), stroke: 2pt + color)
    line((mid + 0.12, y), (right, y), stroke: 1.5pt + color)
  }
  // Josephson junction (cross) on a vertical wire
  let jj(x, y, color) = {
    line((x - 0.3, y + 0.3), (x + 0.3, y - 0.3), stroke: 1.7pt + color)
    line((x + 0.3, y + 0.3), (x - 0.3, y - 0.3), stroke: 1.7pt + color)
  }
  let ground(x, y) = {
    line((x - 0.42, y), (x + 0.42, y), stroke: 1.3pt + slate)
    line((x - 0.27, y - 0.12), (x + 0.27, y - 0.12), stroke: 1.3pt + slate)
    line((x - 0.12, y - 0.24), (x + 0.12, y - 0.24), stroke: 1.3pt + slate)
  }
  let port(x, y) = {
    circle((x, y), radius: 0.11, fill: blue, stroke: none)
    label((x, y + 0.42), [50 Ω port], fill: blue)
  }

  content(
    (0.5, 6.0),
    text(
      size: 8pt,
      font: heading-font,
      weight: "bold",
      fill: blue,
      [TWO JOSEPHSON PARAMETRIC AMPLIFIERS],
    ),
    anchor: "west",
  )
  content(
    (0.5, 5.57),
    text(
      size: 8pt,
      fill: muted,
      [Current pumping modulates $L_upright(J)$ at $2 f_upright(p)$; flux pumping modulates it at $f_upright(p)$.],
    ),
    anchor: "west",
  )

  // ----- Kerr JPA (left) -----
  label(
    (0.5, 4.75),
    [Kerr JPA (four-wave mixing)],
    fill: dark-blue,
    anchor: "west",
  )
  port(0.9, 3.4)
  line((0.9, 3.4), (1.9, 3.4), stroke: 1.5pt + ink)
  hcap(1.9, 3.1, 3.4, light-blue)
  label((2.5, 2.75), $C_upright(c)$, fill: light-blue)
  line((3.1, 3.4), (5.6, 3.4), stroke: 1.5pt + ink)
  circle((4.0, 3.4), radius: 0.07, fill: ink, stroke: none)
  circle((5.6, 3.4), radius: 0.07, fill: ink, stroke: none)
  // junction branch
  line((4.0, 3.4), (4.0, 2.1), stroke: 1.5pt + dark-blue)
  jj(4.0, 2.1, dark-blue)
  line((4.0, 2.1), (4.0, 0.9), stroke: 1.5pt + dark-blue)
  label((3.55, 2.1), $I_upright(c)$, fill: dark-blue, anchor: "east")
  // shunt capacitor branch
  vcap(5.6, 3.4, 0.9, light-blue)
  label((6.15, 2.15), $C$, fill: light-blue, anchor: "west")
  line((4.0, 0.9), (5.6, 0.9), stroke: 1.5pt + ink)
  ground(4.8, 0.9)
  note((0.5, 1.75), [pump $f_upright(p)$], anchor: "west")
  note((0.5, 1.35), [and signal $f_upright(p) + δ$], anchor: "west")
  line((1.0, 2.05), (1.0, 3.15), stroke: 1pt + blue, mark: (
    end: ">",
    size: 0.14,
    fill: blue,
  ))

  line((7.35, 0.6), (7.35, 4.9), stroke: 0.6pt + rule)

  // ----- Flux-pumped CPW JPA (right) -----
  label(
    (7.8, 4.75),
    [Flux-pumped JPA (three-wave mixing)],
    fill: dark-blue,
    anchor: "west",
  )
  port(8.2, 3.4)
  line((8.2, 3.4), (8.9, 3.4), stroke: 1.5pt + ink)
  hcap(8.9, 10.0, 3.4, light-blue)
  label((9.45, 2.75), $C_upright(c)$, fill: light-blue)
  line((10.0, 3.4), (13.4, 3.4), stroke: 3pt + blue)
  label((11.7, 3.85), [λ/4 CPW], fill: blue)
  note((11.7, 2.95), [`tline_ideal` or a rational fit])
  // SQUID to ground at the far end
  line((13.4, 3.4), (14.3, 3.4), stroke: 1.5pt + ink)
  line((14.3, 3.4), (14.3, 2.75), stroke: 1.5pt + dark-blue)
  rect((13.75, 2.75), (14.85, 1.45), stroke: 1.5pt + dark-blue)
  jj(13.75, 2.1, dark-blue)
  jj(14.85, 2.1, dark-blue)
  line((14.3, 1.45), (14.3, 0.9), stroke: 1.5pt + dark-blue)
  ground(14.3, 0.9)
  label((13.4, 2.1), [SQUID], fill: dark-blue, anchor: "east")
  // flux line
  note(
    (10.4, 1.15),
    [$Φ = Φ_"dc" + Φ_upright(p) cos 2π f_upright(p) t$],
    anchor: "center",
  )
  line((12.1, 1.45), (14.15, 2.0), stroke: 1pt + blue, mark: (
    end: ">",
    size: 0.14,
    fill: blue,
  ))
  note((10.4, 0.7), [$f_upright(p) ≈ 2 f_0$])
})
