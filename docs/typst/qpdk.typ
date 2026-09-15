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
  // typsphinx's typst_elements allowlist (papersize, fontsize, lang) cannot
  // carry these, and the tuple title must stay short for the running header,
  // so the full metadata lives here.
  set document(
    title: title,
    author: authors,
    description: "QPDK is an open-source process design kit (PDK) for superconducting quantum RF applications built on gdsfactory: parametric quantum circuit components (transmon qubits, CPW resonators, Josephson junctions, ...), analytical S-parameter models, routing utilities, and test-chip examples.",
    keywords: (
      "QPDK",
      "process design kit",
      "PDK",
      "superconducting quantum circuits",
      "quantum computing",
      "transmon qubits",
      "CPW resonators",
      "Josephson junctions",
      "gdsfactory",
      "GDSII",
      "S-parameters",
      "SAX",
      "JAX",
    ),
  )

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
  // Ragged right like the site; justification rivers badly on the long
  // type-annotation lines the autodoc pages produce.
  set par(justify: false, leading: 0.62em)

  // Headings: Outfit, weights and tracking matching custom.css (700 at h1/h2,
  // 600 deeper down; -0.02em at h1 easing to -0.005em).
  // Headings are always numbered, regardless of `toctree_numbered`: Sphinx's
  // LaTeX builder numbered them too (`:numbered:` is not set on any toctree
  // here, yet the old PDF had numbered sections), and a 400-page reference
  // manual needs them for the outline and cross-references to be usable.
  //
  // The generated index.typ wraps everything under a single level-1 root
  // title.  Rather than numbering it "1" and shifting every real section
  // down, the numbering drops the root's counter step: the root title is
  // unnumbered and level-2 headings read "1", "2", ... like the HTML's top
  // sections.  `outlined: false` keeps it out of the contents page and the
  // PDF bookmarks, so the hierarchy is one level shallower everywhere.
  set heading(numbering: (..nums) => {
    let numbers = nums.pos()
    if numbers.len() < 2 { "" } else { numbering("1.1", ..numbers.slice(1)) }
  })
  show heading.where(level: 1): set heading(outlined: false)
  show heading: it => {
    let sizes = (20pt, 15pt, 12.5pt, 11pt, 10.5pt, 10pt)
    let tracks = (-0.02em, -0.015em, -0.01em, -0.005em, -0.005em, -0.005em)
    let weights = (700, 700, 600, 600, 600, 600)
    // Style by content level, not heading level: level 2 is the top of the
    // real hierarchy after the root title is taken out of the numbering.
    let i = calc.max(calc.min(it.level, 6) - 2, 0)
    set text(
      font: heading-font,
      weight: weights.at(i),
      size: sizes.at(i),
      tracking: tracks.at(i),
      // The site keeps all headings near-black; only the navbar and the
      // rules under the title page carry the accent.
      fill: ink,
    )
    block(above: if it.level == 1 { 1.5em } else { 1.15em }, below: 0.6em, it)
  }

  // Every anchor carries the accent and an underline, matching the site's
  // `a { color: var(--pst-color-link); text-decoration: underline }` (1px
  // stroke, .1578em offset).  Style `it`, never `it.body`: returning the body
  // alone replaces the link element with plain text and the PDF loses its
  // clickable URL annotation.
  show link: it => {
    // Chip links (raw body) skip the underline: drawn around the chip box it
    // lands below the border and reads as a stray blue bar.  The site
    // underlines the code text inside the chip, which Typst cannot reach
    // from here.
    if it.body.func() == raw {
      text(fill: accent, it)
    } else {
      text(
        fill: accent,
        underline(offset: 0.1578em, stroke: 0.75pt + accent, it),
      )
    }
  }

  // Math in Fira Math, matching the HTML MathJax font (mathjax4_config).
  show math.equation: set text(font: math-font)

  show raw: set text(font: mono-font, size: 0.9em)
  show: codly-init.with()
  codly(
    languages: codly-languages,
    zebra-fill: none,
    fill: rgb("#fbfbfc"),
    stroke: 0.5pt + rule,
    number-format: none,
  )
  // Inline code renders as the site's `code.literal` chip: light surface,
  // hairline border, small radius, Bootstrap's violet --bs-code-color.  Must
  // come after codly-init so it styles what codly leaves untouched.
  show raw.where(block: false): it => box(
    fill: rgb("#f3f4f5"),
    stroke: 0.5pt + rgb("#d1d5da"),
    radius: 3pt,
    inset: (x: 3pt, y: 1.2pt),
    text(fill: rgb("#912583"), it),
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
    // Project logo instead of typeset title text, as on the README.  The path
    // resolves against this template copy; docs/conf.py stages the SVG into
    // the bundle because typsphinx does not copy unreferenced _static assets.
    block(width: 100%, stroke: (bottom: 2.5pt + accent), inset: (bottom: 10pt))[
      #image("qpdk_logo.svg", width: 80mm)
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
  // Deliberately NOT `toctree_caption`: typsphinx passes the caption of the
  // *first* toctree in the master document ("API", here), but this outline
  // covers every section, so that caption would mislabel the whole contents
  // page.  The parameter is still accepted because typsphinx always passes it.
  {
    set text(font: heading-font, weight: 700, size: 20pt, tracking: -0.02em)
    block(above: 0pt, below: 0.8em, text(fill: accent)[Contents])
  }
  // Level 2 entries are the top of the contents now that the root title is
  // unoutlined (outlined: false above keeps it out of this query entirely).
  show outline.entry.where(level: 2): it => {
    set text(font: heading-font, weight: 700)
    v(8pt, weak: true)
    it
  }
  outline(title: none, depth: toctree_maxdepth, indent: auto)
  pagebreak()

  body
}
