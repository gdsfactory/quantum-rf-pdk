"""Check the saved hardware comparison used to render the backend comparison in documentation CI."""

from pathlib import Path

import jax.numpy as jnp
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_allclose

RESULT_DIR = Path(__file__).parents[1] / "notebooks/data/jax_backend_comparison"
RUNS = [
    pq.read_table(path).to_pylist()[0] for path in sorted(RESULT_DIR.glob("*.parquet"))
]


def test_saved_backend_coverage() -> None:
    """Keep both accelerators and matching precision baselines in the saved results."""
    backends = {result["backend"] for run in RUNS for result in run["results"]}
    assert backends >= {
        "CPU float64",
        "CPU float32",
        "CUDA float64",
        "CUDA float32",
        "MPS float32",
    }


@pytest.mark.parametrize("run", RUNS, ids=[run["cpu"] for run in RUNS])
def test_saved_run_consistency(run: dict) -> None:
    """Compare the workload, reference trace, and complete timing repetitions."""
    assert run["workload"] == RUNS[0]["workload"]
    trace = run["reference_s21"]
    reference = RUNS[0]["reference_s21"]
    assert len(trace["real"]) == len(trace["imag"]) == 1001
    assert_allclose(trace["real"], reference["real"], rtol=1e-10, atol=1e-12)
    assert_allclose(trace["imag"], reference["imag"], rtol=1e-10, atol=1e-12)
    assert {"CPU float64", "CPU float32"} <= {
        result["backend"] for result in run["results"]
    }
    for result in run["results"]:
        samples = jnp.asarray(result["samples_s"])
        assert samples.shape == (
            len(run["workload"]["sizes"]),
            run["workload"]["n_repeats"],
        )
        assert jnp.all(jnp.isfinite(samples) & (samples > 0))
        assert result["magnitude_error"] < 1e-3
        assert result["complex_error"] < 1e-3
        assert result["output_dtype"] == (
            "complex64" if result["backend"].endswith("float32") else "complex128"
        )
