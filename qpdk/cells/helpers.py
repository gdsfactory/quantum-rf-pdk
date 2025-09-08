"""Helper functions for QPDK cells."""

import gdsfactory as gf
from klayout.db import DCplxTrans


def transform_component(component: gf.Component, transform: DCplxTrans) -> gf.Component:
    """Applies a complex transformation to a component.

    For use with :func:`~gdsfactory.container`.
    """
    return component.transform(transform)
