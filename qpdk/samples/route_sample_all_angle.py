"""Example of routing between two components using all angle routing."""

import gdsfactory as gf

from qpdk import PDK, cells, tech

if __name__ == "__main__":
    PDK.activate()
    c = gf.Component()
    m1 = c << cells.interdigital_capacitor()
    m2 = c << cells.interdigital_capacitor()

    m2.move((400, 200))
    m2.rotate(30)
    route = tech.route_bundle_all_angle(c, [m1.ports["o2"]], [m2.ports["o1"]])
    c.show()
