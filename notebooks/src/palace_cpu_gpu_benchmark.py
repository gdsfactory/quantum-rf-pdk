# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# ---

# %% [markdown]
# # Palace CPU vs GPU Performance
#
# This notebook compares CPU and GPU execution of the driven Palace simulation from the
# `palace_driven_resonator` notebook. Both runs used the identical mesh, geometry, and
# `config.json` produced by gsim there; the only configuration difference is the solver
# device (`"Device": "CPU"` versus `"Device": "GPU"`). The timings reported by Palace are
# embedded below, so the analysis cells run without any Palace installation.
#
# The runs took place on the Aalto Triton cluster:
#
# - **CPU**: 8 MPI ranks on one Cascade Lake node (Xeon Gold 6248, `batch-csl`), 24 GB
#   requested, second-order elements giving about 754k unknowns.
# - **GPU**: 1 MPI rank with one Tesla V100-SXM2-16GB (`gpu-v100-16g`), 16 GB of host
#   memory requested.
#
# Job wall time was 8m20s on CPU and 4m04s on GPU.

# %% tags=["hide-input", "hide-output"]
import sys

if "google.colab" in sys.modules:
    import subprocess

    print("Running in Google Colab. Installing QPDK...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "qpdk[models] @ git+https://github.com/gdsfactory/quantum-rf-pdk.git",
    ])

# %% tags=["hide-input", "hide-output"]
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## The two batch scripts
#
# The CPU job uses the precompiled Palace Apptainer image with the container's own MPI:
#
# ```bash
# #!/bin/bash
# #SBATCH --job-name=palace-cpu
# #SBATCH --partition=batch-milan,batch-csl,batch-skl,batch-bdw,batch-hsw
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=24G
# #SBATCH --time=01:00:00
# #SBATCH --output=/scratch/work/%u/slurm-logs/%x-%j.out
#
# set -euo pipefail
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
#
# SIF=/scratch/work/savolan2/palace-container/palace-x86_64_v3.sif
# cd /scratch/work/savolan2/gsim-palace/sim_qpdk_resonator
# apptainer exec --cleanenv "$SIF" mpirun --oversubscribe -np 8 \
#     palace-x86_64.bin config.json > run.log 2>&1
# ```
#
# The GPU job needs the CUDA-enabled image, `--nv` to expose the GPU inside the container,
# and one MPI rank per GPU. The solver device is switched in the generated configuration:
#
# ```bash
# #!/bin/bash
# #SBATCH --job-name=palace-gpu
# #SBATCH --partition=gpu-v100-16g,gpu-v100-32g
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=16G
# #SBATCH --time=01:00:00
# #SBATCH --output=/scratch/work/%u/slurm-logs/%x-%j.out
#
# set -euo pipefail
#
# SIF=/scratch/work/savolan2/palace-container/palace-cuda.sif
# cd /scratch/work/savolan2/gsim-palace/sim_qpdk_resonator_gpu
# sed -i 's/"Device": "CPU"/"Device": "GPU"/' config.json
# apptainer exec --cleanenv --nv "$SIF" mpirun -np 1 \
#     palace-x86_64.bin config.json > run.log 2>&1
# ```
#
# The same practical notes as in the resonator notebook apply: `--cleanenv` keeps host
# modules out of the container, the raw `palace-x86_64.bin` binary is used instead of the
# `palace` wrapper (which mis-parses the Slurm environment inside an allocation), BLAS
# threading is pinned to one thread per rank, and each run lives in its own
# self-contained directory.

# %% [markdown]
# ## Timings
#
# Palace prints an "Elapsed Time Report" at the end of each run, one column set per MPI
# rank. The numbers below are the per-rank averages of the 8-rank CPU run and the single
# rank of the GPU run. Indented rows are sub-phases of the row above them, so the
# top-level rows do not add up to the total.

# %%
phases = [
    "Initialization",
    "  Mesh Preprocessing",
    "Operator Construction",
    "  Wave Ports",
    "Linear Solve",
    "  Setup",
    "  Preconditioner",
    "  Coarse Solve",
    "PROM Construction",
    "PROM Solve",
    "Estimation",
    "  Construction",
    "  Solve",
    "Postprocessing",
    "  Paraview",
    "Disk IO",
]
cpu_s = np.array([
    3.671,
    1.302,
    42.614,
    0.002,
    23.473,
    56.812,
    222.202,
    18.369,
    1.267,
    2.920,
    0.459,
    6.205,
    69.494,
    47.404,
    0.778,
    1.009,
])
gpu_s = np.array([
    2.243,
    13.031,
    11.845,
    0.001,
    6.807,
    16.324,
    1.892,
    58.648,
    0.206,
    0.092,
    0.314,
    2.992,
    4.877,
    93.974,
    12.346,
    1.302,
])
cpu_total_s = 499.217
gpu_total_s = 228.190
cpu_peak_mem_gb = 6.6
gpu_peak_mem_gb = 2.4

print(f"Total solver time: CPU {cpu_total_s:.1f} s, GPU {gpu_total_s:.1f} s")
print(f"Single V100 vs 8 CPU ranks speedup: {cpu_total_s / gpu_total_s:.1f}x")
print(f"Peak memory: CPU {cpu_peak_mem_gb:.1f} GB, GPU {gpu_peak_mem_gb:.1f} GB")

# %%
y = np.arange(len(phases))
bar_h = 0.4

plt.figure(figsize=(10, 8))
plt.barh(y + bar_h / 2, cpu_s, bar_h, label="CPU (8 ranks, Cascade Lake)")
plt.barh(y - bar_h / 2, gpu_s, bar_h, label="GPU (1x V100)")
plt.yticks(y, phases)
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ## Where the GPU wins and loses
#
# End to end the single V100 finishes about 2.2x faster than the 8 CPU ranks, but the
# phase breakdown shows the speedup is concentrated in the dense linear algebra:
#
# - **Preconditioner** dominates the CPU run at 222.2 s, about 45% of the total, and
#   collapses to 1.9 s on the GPU. Setup (56.8 to 16.3 s), operator construction (42.6 to
#   11.8 s), linear solves (23.5 to 6.8 s), and the reduced-order model's estimation
#   solves (69.5 to 4.9 s) all shrink by a similar factor. These phases are dominated by
#   large sparse factorizations and dense operations that map well onto the GPU.
# - **Coarse Solve** moves the other way, 18.4 to 58.6 s. The coarse-grid correction
#   inside the multigrid-preconditioned solve is a small problem that gains little from
#   the GPU, while the CPU run splits it across 8 ranks.
# - **Mesh Preprocessing** (1.3 to 13.0 s) and **Postprocessing** (47.4 to 94.0 s) are
#   slower on the GPU job for the same reason: they parallelize over MPI ranks, so the
#   single-rank GPU job runs them serially on one host core. The Paraview field output
#   alone takes 12.3 s on the GPU job against 0.8 s on CPU.
#
# Two lessons follow. First, a single GPU replaces a small CPU allocation for this
# problem size, at lower memory as well as lower wall time. The figures above are the
# peak memory Palace reports for each job, aggregated over all ranks: 6.6 GB for the
# 8-rank CPU run (about 0.9 GB per rank, well under the 24 GB request) against 2.4 GB
# for the single-rank GPU run. Second, the GPU advantage grows with the
# linear-algebra share: a bigger mesh, higher element order, or a tighter adaptive
# tolerance makes the preconditioner and solve phases dominate even more, while the
# serial meshing and postprocessing overhead stays roughly fixed.
#
# The two runs converged the adaptive sweep with the same 14 solve samples (greedy error
# indicator 4.2e-05 against a 2e-02 tolerance), and the resulting `port-S.csv` files
# agree to about 1e-9, i.e. single-precision round-off level. The GPU path is not an
# approximation of the CPU result.

# %% [markdown]
# ## Summary
#
# For this driven resonator simulation, one V100 GPU ran Palace about 2.2x faster
# end-to-end than 8 CPU ranks on a comparable cluster node. The win comes entirely from
# the GPU-friendly linear algebra; phases that scale with MPI rank count (mesh
# preprocessing, postprocessing, coarse solves) favor the CPU run. On a cluster the GPU
# queue also matters: V100 partitions are typically the least contended GPU resource,
# so for problems of this size a single-GPU Palace job is the practical default.
