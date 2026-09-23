#let css = read("../_static/css/custom.css")
#let root-match = css.match(regex(":root\\s*\\{([\\s\\S]*?)\\}"))
#let root-css = if root-match == none { "" } else {
  root-match.captures.first()
}
#let css-token(name, pattern, fallback) = {
  let m = root-css.match(regex("--" + name + ":\\s*" + pattern))
  if m == none { fallback } else { m.captures.first() }
}
#let css-color(name, fallback) = rgb(css-token(
  name,
  "(#[0-9a-fA-F]{6})",
  fallback,
))
#let css-font(name, fallback) = {
  let m = css.match(regex("--" + name + ":\\s*\"([^\"]+)\""))
  if m == none { fallback } else { m.captures.first() }
}

#let ink = css-color("qpdk-ink", "#0e1116")
#let blue = css-color("qpdk-accent", "#2a6fb5")
#let light-blue = blue.lighten(7%)
#let dark-blue = css-color("qpdk-accent-hover", "#1f5994")
#let slate = rgb("#5b6472")
#let muted = slate
#let paper = css-color("qpdk-paper", "#f6f4ef")
#let rule = rgb("#dcdfe4")
#let body-font = (css-font("pst-font-family-base", "Inter"), "Arial")
#let heading-font = (css-font("pst-font-family-heading", "Outfit"), "Arial")
