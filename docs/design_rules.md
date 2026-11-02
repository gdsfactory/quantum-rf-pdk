---
myst:
  html_meta:
    description: The QPDK design-rule model, the rules the hosted DRC deck runs, the rule profiles, and the intent layers coupled cells emit.
---

# Design rules

GDSFactory+ runs the QPDK DRC deck for any project set to the QPDK project identifier. This page is the contract with
that deck: what the masks mean, which rules run, how to select a profile, and which layers carry design intent.

The numbers below are what the cells currently draw, not an independent process specification. Confirm them with your
fab before tapeout, and treat a change to the table as a change to the process.

## Mask model

QPDK is a subtractive process. Each metal face has a `*_ETCH` mask that removes a blanket film and a `*_DRAW` mask that
adds metal back, with the additive mask winning where they overlap. The deck therefore does not check the draw mask
directly:

- **Metal width** runs on the blanket film minus the etch mask. Additive metal drawn inside an etched window is fill,
  not a process feature. Nanowire pads and the `open_2_port` port strap are drawn that way.
- **Metal spacing** is the width of the etch gap, `ETCH - DRAW`, because in a subtractive process two pieces of metal
  are separated by an etched slot and nothing else.
- **Optical minimums skip e-beam geometry.** `JJ_AREA`, `NbTiN`, `CNT_CONTACT` and `TOP_GATE` are e-beam layers, and the
  optical minimums ignore everything within 2 µm of them. That is what lets one `M1_ETCH` mask carry 6 µm CPW slots and
  0.09 µm nanowire gaps.

![A blanket niobium film is drawn, two etch windows cut it, and three metal strips remain. Width w is measured on the remaining film, spacing s on the etch slot.](notebooks/figures/drc-mask-model.svg)

## Rules

| Rule       | Check                                                         | Default |
| ---------- | ------------------------------------------------------------- | ------- |
| M1.W.1     | minimum effective M1 width                                    | 2.0 µm  |
| M1.S.1     | minimum M1 metal spacing (etch slot width)                    | 2.0 µm  |
| M2.W.1     | minimum effective top-chip front metal width                  | 2.0 µm  |
| M2.S.1     | minimum top-chip front metal spacing                          | 2.0 µm  |
| MB.W.1     | minimum bottom-chip backside metal width                      | 2.0 µm  |
| MB.S.1     | minimum bottom-chip backside metal spacing                    | 2.0 µm  |
| MB2.W.1    | minimum top-chip backside metal width                         | 2.0 µm  |
| MB2.S.1    | minimum top-chip backside metal spacing                       | 2.0 µm  |
| NBTHIN.W.1 | minimum NbTiN nanowire width                                  | 0.04 µm |
| NBTHIN.S.1 | minimum NbTiN spacing                                         | 0.09 µm |
| JJ.W.1     | minimum `JJ_AREA` width                                       | 0.09 µm |
| JJ.O.1     | `JJ_PATCH` contact overlap along the junction                 | 0.3 µm  |
| AB.E.1     | `AB_VIA` enclosed by effective M1                             | 1.0 µm  |
| AB.O.1     | `AB_VIA` landing pad touches the `AB_DRAW` bridge             | contact |
| AB.C.1     | `AB_VIA` must not land on a CPW centre conductor              | none    |
| TSV.E.1    | TSV enclosed by landing metal on both faces                   | 2.0 µm  |
| IND.E.1    | indium bump enclosed by landing metal on both chips           | 5.0 µm  |
| DICE.K.1   | no drawn metal, via, bump or bridge inside the `DICE` frame   | none    |
| ALN.E.1    | a die frame needs `ALN_TOP` and `ALN_BOT` marks               | present |
| RF.C.1     | unrelated signal lines run closer than the crosstalk distance | 20.0 µm |
| RF.C.2     | a `COUPLING` region covers a real conductor gap               | none    |

Two notes on the junction rules. `JJ_PATCH` is shared with the gatemon process, where it is the Al reconnection film on
a nanotube and has no junction below it, so `JJ.O.1` ignores patches that touch the gatemon layers. The overlap is
measured along the long axis of the intersection, because a QPDK junction wire is 0.2 µm wide and an overlap measured
across the wire can never reach 0.3 µm.

`IND.E.1` and `DICE.K.1` need a chip. They read `SUBSTRATE` or `SUBSTRATE_TOP`, which only a die footprint draws, and
return nothing on a packed cell library like `all_cells`.

## Profiles

The deck selects its rules with the DRC `process` value, which means the same thing as a rule profile here.

| Profile   | Rules                              | Use on                          |
| --------- | ---------------------------------- | ------------------------------- |
| `default` | process geometry and 3D rules      | anything, including `all_cells` |
| `design`  | default plus `RF.C.1` and `RF.C.2` | a chip with coupling markers    |
| `tapeout` | design plus `ALN.E.1`              | the final chip GDS              |

Select a profile per project in `pyproject.toml`:

```toml
[tool.gdsfactoryplus.drc]
timeout = 300
process = "design"
```

or per run with `gfp check drc --process design`. The project identifier still selects the deck, and the process selects
the profile.

## Intent layers

Two non-fabrication layers carry design intent. Both are stripped by `qpdk.utils.remove_metadata_layers`.

- **`WG` (102/0)** marks every CPW centre conductor. `coplanar_waveguide` emits it automatically, so the PDK, not the
  designer, keeps it current. `RF.C.1` measures signal-line spacing on these markers, and merging them first turns each
  connected net into one polygon, so a meander's folds are not mistaken for crosstalk.
- **`COUPLING` (103/0)** marks a corridor where two nets are coupled on purpose. `RF.C.1` waives crosstalk markers
  inside it, and `RF.C.2` reports markers that cover no gap. `RF.C.2` measures the drawn metal masks, not the `WG`
  markers, so a coupling between two plates, or between an island and a line, is valid even though neither part has a
  centre conductor. `qpdk.cells._coupling.mark_coupling_regions` draws the corridor between the two parts from the `gap`
  parameter of the cell that owns both nets. `coupler_straight`, `plate_capacitor` and `unimon_coupled` emit it today.
  Add the call to a composite cell when it couples two elements on purpose, and only around those two elements.

![Two parallel signal lines with the coupling corridor between them. On the left the cell marks the corridor and the deck waives the gap. On the right nothing is marked and the pair is reported.](notebooks/figures/coupling-corridor.svg)

The alternative, hand-drawn waiver polygons, is easy to overuse: one polygon can silence every rule at once. Generated
markers carry the cell's own parameter, cannot be forgotten, and are auditable in review. The deck also keeps a short
list of cells whose geometry is a layout artefact rather than a process feature, currently the `open_2_port` port strap,
which is why it does not appear as an M1 width violation.

## Known DRC findings on the cell library

The `all_cells` run of the `default` profile is not clean today, and the hits are geometry to review, not deck bugs:

- `M1.W.1` reports narrow metal in `neck_termination`, where the smooth window leaves a strip under 2 µm.
- `M1.S.1` reports 2 µm wide wedges where the launcher and `cpw_bulge` tapers cut the etch slot.
- `M2.S.1` reports two slots in the flip-chip transitions.

The `design` profile on `all_cells` also reports `RF.C.1` hits in `coupler_ring`, `tee` and `unimon` variants that are
not marked as coupling. Run it on a chip, where the coupling regions the cells emit are present.

## Changing a rule

Rule implementations live in the hosted deck, not in this repository. Open an issue with the rule, the value you need
and a small layout that fails without it. The deck and the table above change together, and every rule needs a synthetic
case that flips its verdict.
