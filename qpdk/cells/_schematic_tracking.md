# Schematic Symbol Tracking

This file tracks the status of schematic symbols for PDK cells.

| Symbol Name         | Status | Icon / Pattern                             |
| ------------------- | ------ | ------------------------------------------ |
| straight            | ✅     | straight_schematic (\_2PORT)               |
| bend_circular       | ✅     | bend_circular_schematic (\_2PORT)          |
| resonator           | ✅     | resonator_schematic (\_2PORT)              |
| double_pad_transmon | ✅     | double_pad_transmon_schematic (\_TRANSMON) |
| meander_inductor    | ✅     | Packaged inductor SVG (\_2PORT)            |
| half_circle_coupler | ✅     | Single capacitor electrode SVG (o3)        |
| fluxonium           | ❌     | TBD                                        |
| unimon              | ❌     | TBD                                        |
| snspd               | ❌     | TBD                                        |
| inductor            | ❌     | TBD                                        |
| capacitor           | ✅     | Capacitor SVG; single electrode for halves |
| junction            | ✅     | Josephson cross SVG; SQUID pending         |
| launcher            | ❌     | TBD                                        |
| airbridge           | ❌     | TBD                                        |
| bump                | ❌     | TBD                                        |
| tsv                 | ❌     | TBD                                        |
| chip                | ❌     | TBD                                        |

Circuit primitive artwork uses `display.symbol` package resources with the `ckt` type to preserve named QPDK ports. The
editor's native electrical capacitor and inductor symbols use fixed P/N pins. Capacitor electrodes expose only their
actual layout terminals; placement anchors are excluded.

Packaged SVG artwork requires GDSFactory+ support for `display.symbol` resources. Version 2.1.458 does not resolve these
resources; it falls back to layout previews. The metadata retains named ports on either viewer version.
