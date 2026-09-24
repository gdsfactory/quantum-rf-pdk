#let css = read("../_static/css/custom.css")
#let root-match = css.match(regex(":root\\s*\\{([\\s\\S]*?)\\}"))
#let root-css = if root-match == none { "" } else {
  root-match.captures.first()
}
#let css-token(name, pattern) = {
  let m = root-css.match(regex("--" + name + ":\\s*" + pattern))
  if m == none { panic("Missing CSS token: --" + name) } else {
    m.captures.first()
  }
}
#let css-color(name) = rgb(css-token(name, "(#[0-9a-fA-F]{6})"))
#let css-font(name) = {
  let m = css.match(regex("--" + name + ":\\s*\"([^\"]+)\""))
  if m == none { panic("Missing CSS token: --" + name) } else {
    m.captures.first()
  }
}

#let ink = css-color("qpdk-ink")
#let blue = css-color("qpdk-accent")
#let light-blue = blue.lighten(7%)
#let dark-blue = css-color("qpdk-accent-hover")
#let slate = rgb("#5b6472")
#let muted = slate
#let paper = css-color("qpdk-paper")
#let rule = rgb("#dcdfe4")
#let body-font = (css-font("pst-font-family-base"), "Fira Math")
#let heading-font = (css-font("pst-font-family-heading"), "Fira Math")
