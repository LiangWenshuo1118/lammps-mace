# lammps-mace

A MACE/LAMMPS workflow package for molten-salt simulations. It provides one
command-line entry point for building initial boxes, preparing zero-shot
MACE-MD inputs, preparing MACE fine-tuning tasks from DFT-derived data, and
running MD again with a fine-tuned model.

Chinese documentation is available in [README_zh.md](README_zh.md).

After installation, the command is:

```bash
lammps-mace --help
```

## Installation

This section lists the software required by the workflow. The lightweight
Python dependencies of this package are handled by `pip install`. MACE-MD,
fine-tuning, and LAMMPS execution require an external runtime environment.

### Software Versions

The current workflow uses the following versions:

| Software | Version | Used by |
| --- | --- | --- |
| Python | 3.12.x | all stages |
| PyTorch | 2.9.1+cu128 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| MACE | 0.3.16 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| ASE | 3.28.0 | `build-boxes`, `mace-md-npt`, `mace-md-nvt` |
| NumPy | 1.26.4 | all stages |
| CuPy | 13.6.0 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| cuEquivariance | 0.10.0 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| Packmol | conda-forge package | `build-boxes` |
| CMake / Ninja / Open MPI | conda-forge packages | LAMMPS build and execution |
| LAMMPS | develop branch + `ML-IAP` / `PYTHON` / `KOKKOS` | `mace-md-npt`, `mace-md-nvt` |

The example below assumes CUDA 12.8. If your CUDA version or GPU architecture
is different, adjust the PyTorch wheel and the LAMMPS Kokkos architecture flag.

```bash
conda create -y -n mace-lammps-torch291-cu128 -c conda-forge \
  python=3.12 cmake ninja openmpi packmol
conda activate mace-lammps-torch291-cu128

python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128

python -m pip install \
  numpy==1.26.4 "matscipy<1.2" ase==3.28.0 mace-torch==0.3.16 \
  cupy-cuda12x==13.6.0 \
  cuequivariance==0.10.0 cuequivariance-torch==0.10.0 \
  cuequivariance-ops-torch-cu12==0.10.0
```

Compatibility note: when pinning `numpy==1.26.4`, also pin `"matscipy<1.2"`.

### Build LAMMPS ML-IAP from Source

LAMMPS must be built from source to enable the `ML-IAP`, `PYTHON`, and `KOKKOS`
interfaces required by MACE. The commands below assume the target Python
environment is already active and contains PyTorch, MACE, NumPy, CuPy, and
cuEquivariance. `Kokkos_ARCH_ADA89` is an example flag for Ada GPUs.

It is useful to keep the Python environment, source tree, build directory, and
LAMMPS install prefix separate:

```text
/root/base/envs/mace-lammps-torch291-cu128
/root/base/src/lammps-mace
/root/base/src/lammps-develop
/root/base/build/lammps-mliap-torch291-cu128
/root/base/software/lammps-mliap-torch291-cu128
/root/shared-nvme/lammps-mace-demo/outputs
```

```bash
source /path/to/mace-lammps-torch291-cu128/bin/activate

curl -L --fail \
  --output lammps-develop.tar.gz \
  https://github.com/lammps/lammps/archive/refs/heads/develop.tar.gz
tar -xzf lammps-develop.tar.gz
mv lammps-develop-* lammps-develop

cmake -S lammps-develop/cmake -B lammps-build-mliap \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/path/to/lammps-mliap-install \
  -D BUILD_MPI=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_ADA89=ON \
  -D PKG_ML-IAP=ON \
  -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_ML-SNAP=ON \
  -D PKG_PYTHON=ON

cmake --build lammps-build-mliap -j 32
cmake --install lammps-build-mliap
cmake --build lammps-build-mliap --target install-python
```

`cmake --build` compiles the build tree. `cmake --install` installs `lmp` and
`liblammps.so` into the selected prefix. `install-python` installs the matching
`lammps` Python package into the active Python environment so that
`import lammps` works.

Validate the installation:

```bash
LD_LIBRARY_PATH=/path/to/lammps-mliap-install/lib:$LD_LIBRARY_PATH \
  /path/to/lammps-mliap-install/bin/lmp -h

python -c "import lammps; L=lammps.lammps(cmdargs=['-log','none','-screen','none']); print(L.version()); L.close()"
```

### Install This Package

This package requires Python 3.10 or newer. For local development:

```bash
python -m pip install -e .
```

To install from GitHub:

```bash
python -m pip install git+https://github.com/LiangWenshuo1118/lammps-mace.git
```

### Validate the Python Tools

```bash
python -c "import ase; print(ase.__version__)"
packmol -h
lammps-mace --help
```

### Base Model Files

Place the MACE-MP-0b3-medium checkpoint under `models/base/`. This directory is
intended for large model files needed by local or remote runs. LAMMPS ML-IAP
cannot directly read a `.model` checkpoint, so convert it to a
`*-mliap_lammps.pt` file before running zero-shot MACE-MD.

## Quick Start

The example below is a complete minimal workflow. It assumes the Python
environment is active and the LAMMPS ML-IAP install prefix is available.

### 1. Initialize a Project and Build Boxes

```bash
lammps-mace init --target demo
cd demo

lammps-mace build-boxes \
  --csv examples/systems.csv \
  --output-root outputs \
  --density 1.46
```

`examples/systems.csv` is the initial system table. Common columns are
`system_index`, `salts`, `concentrations`, `concentration_type`, and
`total_atoms`. Box density is not stored in the CSV; it is provided by
`build-boxes --density`. Box-building outputs are written under each system's
`simulation_box/` directory. `--packmol-tolerance` is optional and defaults to
`2.0`.

### 2. Prepare the Base Model

Put the base MACE `.model` file under `models/base/`, then convert it to the
LAMMPS ML-IAP `.pt` format:

```bash
mkdir -p models/base
cp /path/to/mace-mp-0b3-medium.model models/base/

python -m mace.cli.create_lammps_model \
  models/base/mace-mp-0b3-medium.model \
  --format=mliap
```

The converted file is usually:

```text
models/base/mace-mp-0b3-medium.model-mliap_lammps.pt
```

### 3. Generate and Submit Base-Model NPT Tasks

```bash
lammps-mace mace-md-npt \
  --boxes-root outputs \
  --model-path models/base/mace-mp-0b3-medium.model-mliap_lammps.pt \
  --temperature 973 1073 \
  --pressure 0 1 \
  --npt-steps 1000 \
  --timestep-fs 1.0 \
  --thermo-every 100 \
  --dump-every 100

bash outputs/run_mace_md_npt_all.sh
```

`--temperature` and `--pressure` both accept multiple values. The command
creates the Cartesian product of temperature and pressure conditions, for
example:

```text
outputs/run_mace_md_npt_all.sh
outputs/001_LiCl/973K_00bar/mace-md-npt/run.sh
outputs/001_LiCl/1073K_00bar/mace-md-npt/run.sh
```

The generated run scripts default to:

```text
/root/base/envs/mace-lammps-torch291-cu128
/root/base/software/lammps-mliap-torch291-cu128
```

If your paths differ, set these variables before submitting:

```bash
export MACE_ENV=/path/to/mace-lammps-torch291-cu128
export LAMMPS_PREFIX=/path/to/lammps-mliap-install
```

Each successful task writes an empty `task_finished` marker. If a run is
interrupted, rerunning the aggregate script skips directories that already have
this marker.

### 4. Generate and Submit Base-Model NVT Tasks

```bash
lammps-mace mace-md-nvt \
  --boxes-root outputs \
  --model-path models/base/mace-mp-0b3-medium.model-mliap_lammps.pt \
  --nvt-steps 1000 \
  --timestep-fs 1.0 \
  --thermo-every 100 \
  --dump-every 100

bash outputs/run_mace_md_nvt_all.sh
```

`mace-md-nvt` follows existing NPT condition directories. By default, NVT reads
`simulation_box/box.data`, so it does not require the NPT runs to be complete.
To generate NVT for selected temperatures only, add `--temperature 973`. To size
NVT boxes from the late-stage NPT average density, add `--use-npt-density 0.8`.
In that mode, each corresponding NPT directory must already contain
`final_mace_npt.data` and `log.kk.half.lammps`.

### 5. Prepare and Run MACE Fine-Tuning

Fine-tuning data are extxyz files. This example assumes:

```text
data/finetune/Training.extxyz
data/finetune/Validation.extxyz
```

```bash
lammps-mace finetune \
  --task-dir outputs/finetune-mace \
  --dataset-root data/finetune \
  --train-file Training.extxyz \
  --valid-file Validation.extxyz \
  --foundation-model models/base/mace-mp-0b3-medium.model \
  --name mace-mp-0b3-medium-finetuned \
  --max-num-epochs 6 \
  --batch-size 2 \
  --forces-key force \
  --stress-key stress \
  --stress-weight 1.0 \
  --e0s average \
  --energy-weight 1.0 \
  --forces-weight 1.0 \
  --lr 0.01 \
  --scaling rms_forces_scaling \
  --ema \
  --ema-decay 0.99 \
  --amsgrad \
  --default-dtype float64 \
  --seed 3 \
  --no-multiheads-finetuning \
  --smoke-max-configs 100 \
  --require-dataset

cd outputs/finetune-mace
nohup bash run_train.sh > nohup.out 2>&1 &
```

This creates:

```text
outputs/finetune-mace/
├── checkpoints/
├── logs/
├── models/
├── results/
├── finetune_settings.json
└── run_train.sh
```

`--smoke-max-configs` is optional and is only intended for smoke tests. Remove it
for full training runs.

### 6. Generate MD Tasks with the Fine-Tuned Model

The previous step submits training from inside `outputs/finetune-mace`. After
training finishes, return to the project root:

```bash
cd ../..
```

```bash
python -m mace.cli.create_lammps_model \
  outputs/finetune-mace/models/mace-mp-0b3-medium-finetuned.model \
  --format=mliap

lammps-mace mace-md-npt \
  --boxes-root outputs \
  --model-path outputs/finetune-mace/models/mace-mp-0b3-medium-finetuned.model-mliap_lammps.pt \
  --task-suffix finetuned \
  --temperature 973 \
  --pressure 0 \
  --npt-steps 100 \
  --timestep-fs 1.0 \
  --thermo-every 10

bash outputs/run_mace_md_npt_finetuned_all.sh

lammps-mace mace-md-nvt \
  --boxes-root outputs \
  --model-path outputs/finetune-mace/models/mace-mp-0b3-medium-finetuned.model-mliap_lammps.pt \
  --task-suffix finetuned \
  --nvt-steps 100 \
  --timestep-fs 1.0 \
  --thermo-every 10 \
  --use-npt-density 0.8

bash outputs/run_mace_md_nvt_finetuned_all.sh
```

`--task-suffix finetuned` creates `mace-md-npt-finetuned` and
`mace-md-nvt-finetuned`, avoiding overwrites of the base-model tasks. A
fine-tuned model usually supports only the elements present in the fine-tuning
data. If a system contains unsupported elements, LAMMPS may fail at the
`pair_coeff` stage.

## References

- [Python venv](https://docs.python.org/3.12/tutorial/venv.html)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [MACE PyPI](https://pypi.org/project/mace-torch/)
- [NVIDIA cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/)
- [CuPy](https://docs.cupy.dev/en/v13.6.0/install.html)
- [MACE ML-IAP](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)
- [LAMMPS Python](https://docs.lammps.org/Python_install.html)
- [Packmol conda-forge](https://anaconda.org/conda-forge/packmol)
- [Open MPI conda-forge](https://anaconda.org/conda-forge/openmpi)
