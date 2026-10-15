# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
# ]
# ///
"""Record synchronized CPU, CUDA, or MPS circuit timings to a Parquet file."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sax
from tqdm import tqdm

from qpdk import PDK
from qpdk.config import PATH
from qpdk.models.generic import capacitor, tee
from qpdk.models.waveguides import straight, straight_shorted
from qpdk.tech import coplanar_waveguide


def _main() -> None:
    """Benchmark available devices and save raw timings with their provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Parquet output path")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_device", jax.devices("cpu")[0])

    PDK.activate()

    cpu_device = jax.devices("cpu")[0]

    try:
        gpu_device = jax.devices("gpu")[0]
    except RuntimeError:
        gpu_device = None
    HAS_GPU = gpu_device is not None

    if os.environ.get("QPDK_REQUIRE_GPU") == "1" and not HAS_GPU:
        raise RuntimeError(
            "QPDK_REQUIRE_GPU=1 but JAX found no GPU. Install a CUDA build of JAX, "
            'e.g. `pip install "jax[cuda12]"`, and check `nvidia-smi`.'
        )

    try:
        mps_device = jax.devices("mps")[0]
    except RuntimeError:
        mps_device = None
    HAS_MPS = mps_device is not None

    if os.environ.get("QPDK_REQUIRE_MPS") == "1" and not HAS_MPS:
        raise RuntimeError(
            "QPDK_REQUIRE_MPS=1 but JAX found no MPS device. "
            "Install jax-mps after SAX supports JAX 0.11."
        )

    cpu_kind = platform.processor()
    if platform.system() == "Linux":
        cpu_kind = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        )
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    if platform.system() == "Darwin":
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        cpu_kind = chip
        print(f"Chip : {chip}; macOS {platform.mac_ver()[0]}")
    else:
        print(f"CPU model: {cpu_kind}")
    print(f"Python: {platform.python_version()}")
    print(
        f"JAX  : {jax.__version__}; jaxlib {version('jaxlib')}; SAX {sax.__version__}"
    )
    print(f"CPU  : ✓  ({cpu_device.device_kind})")
    print(
        f"GPU  : {'✓  (' + gpu_device.device_kind + ')' if HAS_GPU else '✗  (not available)'}"
    )
    print(
        f"MPS  : {'available (' + mps_device.device_kind + ')' if HAS_MPS else 'not available'}"
    )
    if HAS_MPS:
        print(f"jax-mps: {version('jax-mps')}")
        print(f"JAX_MPS_ASYNC_DISPATCH={os.environ.get('JAX_MPS_ASYNC_DISPATCH', '0')}")
    print(f"\nDefault JAX backend: {jax.default_backend()}")

    cross_section = coplanar_waveguide(width=10, gap=6)

    circuit_models = {
        "straight": straight,
        "capacitor": capacitor,
        "straight_shorted": straight_shorted,
        "tee": tee,
    }

    netlist = {
        "instances": {
            "feedline1": {
                "component": "straight",
                "settings": {"length": 500, "cross_section": cross_section},
            },
            "feedline2": {
                "component": "straight",
                "settings": {"length": 500, "cross_section": cross_section},
            },
            "cap": {
                "component": "capacitor",
                "settings": {"capacitance": 20e-15, "z0": 50},
            },
            "res": {
                "component": "straight_shorted",
                "settings": {"length": 4000, "cross_section": cross_section},
            },
            "tee": "tee",
        },
        "connections": {
            "feedline1,o2": "tee,o1",
            "tee,o2": "feedline2,o1",
            "tee,o3": "cap,o1",
            "cap,o2": "res,o1",
        },
        "ports": {
            "o1": "feedline1,o1",
            "o2": "feedline2,o2",
        },
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        circuit_fn, _ = sax.circuit(
            netlist=netlist, models=circuit_models, backend="filipsson_gunnar"
        )

    print("Circuit built successfully.")
    print("External ports:", list(netlist["ports"].keys()))

    frequency_range = (2e9, 8e9)
    freq_ref = jnp.linspace(*frequency_range, 1001, dtype=jnp.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_ref = circuit_fn(f=freq_ref)

    s21_ref = s_ref["o1", "o2"]

    circuit_jit = jax.jit(circuit_fn)

    benchmark_sizes = np.geomspace(100, 30_000_000, 19, dtype=int).tolist()
    n_repeats = 10

    def benchmark_circuit(
        device: jax.Device,
        sizes: list[int] = benchmark_sizes,
        n_repeats: int = n_repeats,
        *,
        single_precision: bool = False,
    ) -> list[list[float]]:
        """Return raw wall-clock samples [s] per circuit evaluation on *device*.

        Args:
            device: JAX device on which to run the circuit.
            sizes: List of frequency-array lengths to sweep.
            n_repeats: Number of timed repetitions per size.
            single_precision: Use float32 inputs and complex64 arithmetic.

        Returns:
            Timed repetitions, grouped by frequency-array size.

        Raises:
            RuntimeError: If the circuit runs at the wrong precision or on the wrong device.
        """
        samples: list[list[float]] = []
        with jax.enable_x64(not single_precision), jax.default_device(device):
            dtype = jnp.float32 if single_precision else jnp.float64
            for n in tqdm(sizes, desc=f"Benchmarking on {device}, {dtype}"):
                freq = jax.device_put(
                    jnp.linspace(*frequency_range, n, dtype=dtype), device
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = jax.block_until_ready(circuit_jit(f=freq))
                    expected_dtype = (
                        jnp.complex64 if single_precision else jnp.complex128
                    )
                    if any(
                        value.dtype != expected_dtype or value.device != device
                        for value in result.values()
                    ):
                        raise RuntimeError(
                            "Circuit output precision or device is incorrect."
                        )

                    del result
                    run_times: list[float] = []
                    for _ in range(n_repeats):
                        t0 = time.perf_counter()
                        jax.block_until_ready(circuit_jit(f=freq))
                        run_times.append(time.perf_counter() - t0)
                median = float(np.median(run_times))
                spread = float(np.std(run_times, ddof=1))
                samples.append(run_times)
                tqdm.write(
                    f"  n={n:>6d}: median {median * 1_000:.3f} ms, std {spread * 1_000:.3f} ms"
                )
        return samples

    cpu_samples = benchmark_circuit(cpu_device)
    cpu_float32_samples = benchmark_circuit(cpu_device, single_precision=True)

    if HAS_GPU:
        gpu_samples = benchmark_circuit(gpu_device)
        gpu_float32_samples = benchmark_circuit(gpu_device, single_precision=True)
    else:
        print("CUDA unavailable; recording CPU measurements only.")

    if HAS_MPS:
        mps_samples = benchmark_circuit(mps_device, single_precision=True)
    else:
        print("MPS not available on this system; skipping Apple Silicon benchmark.")

    backend_devices = {
        "CPU float64": (cpu_device, False),
        "CPU float32": (cpu_device, True),
    }
    if HAS_GPU:
        backend_devices["CUDA float64"] = (gpu_device, False)
        backend_devices["CUDA float32"] = (gpu_device, True)
    if HAS_MPS:
        backend_devices["MPS float32"] = (mps_device, True)

    reference = np.asarray(s21_ref)
    accuracy = {}
    for name, (device, single_precision) in backend_devices.items():
        with jax.enable_x64(not single_precision), jax.default_device(device):
            freq = jax.device_put(
                jnp.asarray(
                    np.asarray(freq_ref),
                    dtype=jnp.float32 if single_precision else jnp.float64,
                ),
                device,
            )
            result = np.asarray(jax.block_until_ready(circuit_jit(f=freq))["o1", "o2"])
        if not np.isfinite(result).all():
            raise RuntimeError(f"{name} produced non-finite transmission.")
        magnitude_error = np.max(np.abs(np.abs(result) - np.abs(reference)))
        complex_error = np.max(np.abs(result - reference))
        accuracy[name] = {
            "output_dtype": str(result.dtype),
            "magnitude_error": float(magnitude_error),
            "complex_error": float(complex_error),
        }

    local_samples = {"CPU float64": cpu_samples, "CPU float32": cpu_float32_samples}
    if HAS_GPU:
        local_samples["CUDA float64"] = gpu_samples
        local_samples["CUDA float32"] = gpu_float32_samples
    if HAS_MPS:
        local_samples["MPS float32"] = mps_samples

    workload = {
        "sizes": benchmark_sizes,
        "n_repeats": n_repeats,
        "frequency_hz": list(frequency_range),
        "solver": "filipsson_gunnar",
        "netlist": json.loads(
            json.dumps(netlist, default=lambda value: value.model_dump(mode="json"))
        ),
        "model_sha256": hashlib.sha256(
            b"".join(
                (PATH.module / name).read_bytes()
                for name in (
                    "models/waveguides.py",
                    "models/cpw.py",
                    "models/generic.py",
                    "models/constants.py",
                    "tech.py",
                )
            )
        ).hexdigest(),
    }
    local_run = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "system": platform.system(),
        "cpu": cpu_kind,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("jax", "jaxlib", "sax", "jaxellip")
        },
        "workload": workload,
        "reference_s21": {
            "real": reference.real.tolist(),
            "imag": reference.imag.tolist(),
        },
        "results": [],
    }
    if HAS_MPS:
        local_run["versions"]["jax-mps"] = version("jax-mps")
        local_run["mps_async_dispatch"] = os.environ.get("JAX_MPS_ASYNC_DISPATCH", "0")
    for name, samples in local_samples.items():
        device = backend_devices[name][0]
        local_run["results"].append({
            "backend": name,
            "hardware": cpu_kind
            if device.platform == "cpu"
            else (
                cpu_kind + " GPU" if device.platform == "mps" else device.device_kind
            ),
            "samples_s": samples,
            **accuracy[name],
        })

    pq.write_table(pa.Table.from_pylist([local_run]), args.output, compression="zstd")


if __name__ == "__main__":
    _main()
