"""Tests for analytical ECCPW mutual capacitance formula."""

import jax.numpy as jnp
from hypothesis import assume, given, settings, strategies as st

from qpdk.models.couplers import cpw_cpw_coupling_capacitance_per_length_analytical

# Bounds are set to physically reasonable values to avoid numerical issues
permittivities = st.floats(min_value=1.0, max_value=20.0)


def is_valid_geometry(gap: float, width: float, cpw_gap: float) -> bool:
    """Check if the geometry is within the valid domain for conformal mapping."""
    x1 = gap / 2
    x2 = x1 + width
    x3 = x2 + cpw_gap
    ko_sq = (x1**2 / x2**2) * ((x3**2 - x2**2) / (x3**2 - x1**2))
    return float(ko_sq) < 0.95


@st.composite
def valid_cpw_geometry(draw: st.DrawFn) -> tuple[float, float, float]:
    """Generate a valid combination of (gap, width, cpw_gap)."""
    gap = draw(st.floats(min_value=0.1, max_value=10.0))
    width = draw(st.floats(min_value=1.0, max_value=50.0))
    cpw_gap = draw(st.floats(min_value=1.0, max_value=50.0))

    assume(is_valid_geometry(gap, width, cpw_gap))

    return gap, width, cpw_gap


@st.composite
def valid_monotonic_geometries(draw: st.DrawFn) -> tuple[float, float, float, float]:
    """Generate a valid combination of (gap1, gap2, width, cpw_gap) for monotonicity testing."""
    gap1 = draw(st.floats(min_value=0.1, max_value=5.0))
    gap_diff = draw(st.floats(min_value=0.1, max_value=5.0))
    gap2 = gap1 + gap_diff
    width = draw(st.floats(min_value=1.0, max_value=50.0))
    cpw_gap = draw(st.floats(min_value=1.0, max_value=50.0))

    assume(is_valid_geometry(gap1, width, cpw_gap))
    assume(is_valid_geometry(gap2, width, cpw_gap))

    return gap1, gap2, width, cpw_gap


@settings(deadline=None)
@given(
    geometry=valid_cpw_geometry(),
    ep_r=permittivities,
)
def test_cpw_cpw_coupling_capacitance_positivity(
    geometry: tuple[float, float, float], ep_r: float
) -> None:
    """Verify the formula returns a positive value for all valid CPW geometries."""
    gap, width, cpw_gap = geometry
    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )
    assert not jnp.isnan(c_pul)
    assert float(c_pul) > 0


@settings(deadline=None)
@given(
    geometries=valid_monotonic_geometries(),
    ep_r=permittivities,
)
def test_cpw_cpw_coupling_capacitance_monotonicity(
    geometries: tuple[float, float, float, float], ep_r: float
) -> None:
    """Verify coupling capacitance behavior as inter-CPW gap increases."""
    gap1, gap2, width, cpw_gap = geometries

    c_pul1 = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap1, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )
    c_pul2 = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap2, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )

    # Capacitance per unit length decreases as the physical gap increases.
    assert float(c_pul1) > float(c_pul2)


def test_cpw_cpw_coupling_capacitance_consistency() -> None:
    """Check consistency with expected values for a known geometry.

    For gap=0.27 µm, width=10 µm, cpw_gap=6 µm, ep_r=11.7,
    we expect a specific calculated output from the analytical model per unit length.
    """
    gap = 0.27
    width = 10.0
    cpw_gap = 6.0
    ep_r = 11.7

    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )

    # Convert to femtoFarads per meter
    c_pul_ff_m = float(c_pul) * 1e15

    # Check that it falls within the expected physical range (around 1.5e5 fF/m)
    assert 1e4 < c_pul_ff_m < 1e6


@settings(deadline=None)
@given(
    geometry=valid_cpw_geometry(),
    ep_r=permittivities,
)
def test_cpw_cpw_coupling_capacitance_permittivity_scaling(
    geometry: tuple[float, float, float], ep_r: float
) -> None:
    """Verify that capacitance scales with effective permittivity (ep_r + 1)."""
    gap, width, cpw_gap = geometry

    c_pul = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap, width=width, cpw_gap=cpw_gap, ep_r=ep_r
    )

    # Reference with vacuum (ep_r = 1.0)
    c_pul_vacuum = cpw_cpw_coupling_capacitance_per_length_analytical(
        gap=gap, width=width, cpw_gap=cpw_gap, ep_r=1.0
    )

    # ep_eff = (ep_r + 1) / 2
    # c_m should be proportional to ep_eff
    ep_eff_ratio = ((ep_r + 1) / 2) / ((1.0 + 1) / 2)
    assert jnp.isclose(c_pul, float(c_pul_vacuum) * ep_eff_ratio, rtol=1e-5)
