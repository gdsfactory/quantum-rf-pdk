// Layer colours and GDS numbers read from qpdk/layers.yaml, so figures draw
// each layer the way KLayout shows it and never carry a hand-copied hex code.
// Files whose name starts with `_` are imported by figures, not built as one.

#let _flatten(views) = {
  let out = (:)
  for (name, view) in views {
    if "group_members" in view { out += _flatten(view.group_members) }
    if "layer" in view { out.insert(name, view) }
  }
  out
}

#let layer-views = _flatten(yaml("../../qpdk/layers.yaml").LayerViews)

#let layer-view(name) = {
  assert(name in layer-views, message: "not in qpdk/layers.yaml: " + name)
  layer-views.at(name)
}

// A view sets `color`, or `fill_color` when only its fill is coloured.
#let layer-color(name) = {
  let view = layer-view(name)
  rgb(view.at("fill_color", default: view.at("color", default: none)))
}

// "M1_DRAW (1/0)"
#let layer-label(name) = {
  let (layer, datatype) = layer-view(name).layer
  name + " (" + str(layer) + "/" + str(datatype) + ")"
}

// Outline for a drawn layer: its _ETCH partner when it has one, so a metal
// reads like the layout, else a darker shade of the fill.
#let layer-frame(name) = {
  let etch = name.replace(regex("_DRAW$"), "_ETCH")
  if etch != name and etch in layer-views { layer-color(etch) } else {
    layer-color(name).darken(35%)
  }
}
