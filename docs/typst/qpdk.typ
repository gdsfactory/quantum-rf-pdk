// QPDK brand template for the Typst PDF documentation.
//
// Echoes the HTML theme (docs/_static/css/custom.css): Outfit for headings at
// 700 with tightened tracking, Inter for body text, Code New Roman for code,
// and #2a6fb5 as the accent.  The page is pure white rather than the site's
// warm --qpdk-paper (#f6f4ef): that off-white reads as a deliberate screen
// surface, but in print it looks like a scanning artefact, and a tinted
// background costs ink on every page of a 400-page manual.
//
// The `project` signature is fixed by typsphinx: it passes title, authors,
// date, toctree_* (from the master toctree) and the `typst_elements` keys.

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *

#let accent = rgb("#2a6fb5")
#let ink = rgb("#0e1116")
#let muted = rgb("#5b6472")
#let rule = rgb("#dcdfe4")

#let heading-font = ("Outfit", "DejaVu Sans")
#let body-font = ("Inter", "DejaVu Sans")
#let mono-font = ("Code New Roman", "DejaVu Sans Mono")
#let math-font = ("Fira Math", "New Computer Modern Math")

#let project(
  title: "",
  authors: (),
  date: none,
  toctree_maxdepth: 2,
  toctree_numbered: false,
  toctree_caption: "Contents",
  papersize: "a4",
  fontsize: 10pt,
  lang: "en",
  body,
) = {
  set document(title: title, author: authors)

  set page(
    paper: papersize,
    margin: (top: 26mm, bottom: 24mm, inside: 24mm, outside: 22mm),
    numbering: "1",
    number-align: center,
    header: context {
      if counter(page).get().first() <= 1 { return }
      // Every document sits under the root doc's single level-1 heading, so
      // the level-2 headings are the real sections ("PCells", "Models", ...).
      let seen = query(selector(heading.where(level: 2)).before(here()))
      set text(font: heading-font, size: 8pt, fill: muted)
      block(width: 100%, stroke: (bottom: 0.5pt + rule), inset: (bottom: 4pt))[
        #upper(title) #h(1fr) #if seen.len() > 0 { seen.last().body }
      ]
    },
  )

  set text(font: body-font, size: fontsize, lang: lang, fill: ink)
  set par(justify: true, leading: 0.62em)

  // Headings: Outfit 700, tracking tightened the way custom.css does
  // (-0.02em at h1 easing to -0.005em deeper down).
  set heading(numbering: if toctree_numbered { "1.1" } else { "1.1" })
  show heading: it => {
    let sizes = (20pt, 15pt, 12.5pt, 11pt, 10.5pt, 10pt)
    let tracks = (-0.02em, -0.015em, -0.01em, -0.005em, -0.005em, -0.005em)
    let i = calc.min(it.level, 6) - 1
    set text(
      font: heading-font,
      weight: 700,
      size: sizes.at(i),
      tracking: tracks.at(i),
      fill: if it.level == 1 { accent } else { ink },
    )
    block(above: if it.level == 1 { 1.5em } else { 1.15em }, below: 0.6em, it)
  }

  // Links carry the accent; internal cross-references stay inked so the
  // body does not turn blue (`it.dest` is a str only for external URLs).
  show link: it => if type(it.dest) == str { text(fill: accent, it.body) } else { it }

  // Math in Fira Math, matching the HTML MathJax font (mathjax4_config).
  show math.equation: set text(font: math-font)

  show raw: set text(font: mono-font, size: 0.92em)
  show: codly-init.with()
  codly(
    languages: codly-languages,
    zebra-fill: none,
    fill: rgb("#fbfbfc"),
    stroke: 0.5pt + rule,
    number-format: none,
  )

  // Figures and tables
  show figure.caption: set text(size: 0.88em, fill: muted)
  set table(stroke: (x, y) => (
    top: if y == 0 { 0.8pt + ink } else if y == 1 { 0.5pt + rule } else { 0pt },
    bottom: 0.8pt + ink,
  ))
  show table.cell.where(y: 0): set text(font: heading-font, weight: 700)

  // ---- Title page -------------------------------------------------------
  page(header: none, numbering: none, {
    v(52mm)
    block(width: 100%, stroke: (bottom: 2.5pt + accent), inset: (bottom: 10pt))[
      #set text(font: heading-font, weight: 700, size: 40pt, tracking: -0.02em)
      #text(fill: accent)[#title]
    ]
    v(6pt)
    set text(font: heading-font, size: 11.5pt, fill: muted)
    block[Superconducting Quantum Process Design Kit]
    v(1fr)
    set text(font: body-font, size: 10pt, fill: muted)
    block[#authors.join(", ")]
    if date != none and date != "" { block[#date] }
  })
  counter(page).update(1)

  // ---- Contents ---------------------------------------------------------
  if toctree_caption != "" {
    set text(font: heading-font, weight: 700, size: 20pt, tracking: -0.02em)
    block(above: 0pt, below: 0.8em, text(fill: accent)[#toctree_caption])
  }
  show outline.entry.where(level: 1): it => {
    set text(font: heading-font, weight: 700)
    v(8pt, weak: true)
    it
  }
  outline(title: none, depth: toctree_maxdepth, indent: auto)
  pagebreak()

  body
}
