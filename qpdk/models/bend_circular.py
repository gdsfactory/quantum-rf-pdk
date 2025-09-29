"""S-parameter model for a circular bend."""

from typing import Unpack

import sax

from qpdk.models.straight import StraightModelKwargs, straight


def bend_circular(**kwargs: Unpack[StraightModelKwargs]) -> sax.SType:
    """S-parameter model for a circular bend.

    This is wrapped to :func:`~straight`.

    Returns:
        sax.SType: S-parameters dictionary
    """
    return straight(**kwargs)
