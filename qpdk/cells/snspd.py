"""Superconducting nanowire single-photon detector (SNSPD)."""

from functools import partial

import gdsfactory.components.superconductors.snspd

from qpdk import tech

snspd = partial(gdsfactory.components.superconductors.snspd, layer=tech.LAYER.M1_DRAW)


if __name__ == "__main__":
    from qpdk import PDK

    PDK.activate()

    c = snspd()
    c.show()
