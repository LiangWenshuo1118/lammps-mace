# lammps-mace

一个 MACE/LAMMPS 工作流包，目标是把熔盐体系从构盒、零样本 MACE-MD、DFT 数据微调，到微调模型 MACE-MD 的流程收敛到统一命令行入口。当前包名为 `lammps-mace`，安装后提供命令：

English documentation is available in [README.md](README.md).

项目仓库：[https://github.com/LiangWenshuo1118/lammps-mace](https://github.com/LiangWenshuo1118/lammps-mace)

```bash
lammps-mace --help
```

## 安装

本节记录运行本仓库工作流所需的软件。本包自身的轻量 Python 依赖会随 `pip install` 自动处理；MACE-MD、微调和 LAMMPS 执行环境依赖需要用户在运行环境中额外安装。

### 软件版本

当前工作流使用下面这组版本：

| 软件 | 版本 | 使用阶段 |
| --- | --- | --- |
| Python | 3.12.x | 全部阶段 |
| PyTorch | 2.9.1+cu128 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| MACE | 0.3.16 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| ASE | 3.28.0 | `build-boxes`, `mace-md-npt`, `mace-md-nvt` |
| NumPy | 1.26.4 | 全部阶段 |
| CuPy | 13.6.0 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| cuEquivariance | 0.10.0 | `mace-md-npt`, `mace-md-nvt`, `finetune` |
| Packmol | conda-forge package | `build-boxes` |
| CMake / Ninja / Open MPI | conda-forge packages | LAMMPS 编译与运行 |
| LAMMPS | develop branch + `ML-IAP` / `PYTHON` / `KOKKOS` | `mace-md-npt`, `mace-md-nvt` |

下面以 CUDA 12.8 环境为例。若本机 CUDA 或 GPU 架构不同，请相应调整 PyTorch wheel 和后面的 LAMMPS Kokkos 架构选项。

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

兼容性备注：固定 `numpy==1.26.4` 时，建议同时固定 `"matscipy<1.2"`。

### LAMMPS ML-IAP 源码安装

LAMMPS 需要源码编译，以启用 MACE 所需的 `ML-IAP`、`PYTHON` 和 `KOKKOS`接口。下面命令假设已经激活目标 Python venv，并且该 venv 里已经装好 PyTorch、MACE、NumPy、CuPy 和 cuEquivariance。`Kokkos_ARCH_ADA89` 是示例配置，适用于 Ada 架构 GPU。

推荐把 Python 环境、源码、编译目录和 LAMMPS 安装前缀分开管理，例如：

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

`cmake --build` 只在 build 目录编译产物；`cmake --install` 把 `lmp` 和 `liblammps.so` 安装到指定前缀；`install-python` 会把与本次编译匹配的`lammps` Python 包安装进当前 venv，使 `import lammps` 可用。

安装后验证：

```bash
LD_LIBRARY_PATH=/path/to/lammps-mliap-install/lib:$LD_LIBRARY_PATH \
  /path/to/lammps-mliap-install/bin/lmp -h

python -c "import lammps; L=lammps.lammps(cmdargs=['-log','none','-screen','none']); print(L.version()); L.close()"
```

### 安装本包

本包要求 Python 3.10+。本地开发安装：

```bash
python -m pip install -e .
```

从 GitHub 安装时可使用：

```bash
python -m pip install git+https://github.com/LiangWenshuo1118/lammps-mace.git
```

### 验证安装

```bash
python -c "import ase; print(ase.__version__)"
packmol -h
lammps-mace --help
```

### 基础模型文件

MACE-MP-0b3-medium 的基础 checkpoint 放在 `models/base/`。该目录适合存放本地或远端运行所需的大模型文件。LAMMPS ML-IAP 零样本 MD 需要先把 `.model` checkpoint 转换为 `*-mliap_lammps.pt` 文件。

## 快速开始

下面是一套完整的最小使用案例。假设已经激活 Python 环境，并且 LAMMPS ML-IAP 安装前缀可用。

### 1. 初始化项目并构建模拟盒子

```bash
lammps-mace init --target demo
cd demo

lammps-mace build-boxes \
  --csv examples/systems.csv \
  --output-root outputs \
  --density 1.46
```

`examples/systems.csv` 是初始体系表，常用列为 `system_index`、`salts`、`concentrations`、`concentration_type` 和 `total_atoms`。构盒密度不写在 CSV 中，由 `build-boxes --density` 指定。构盒结果会写入每个体系目录下的 `simulation_box/`；`--packmol-tolerance` 可选，默认值为 `2.0`。

### 2. 准备基础模型

将基础 MACE `.model` 放入 `models/base/`，并转换为 LAMMPS ML-IAP 可读的 `.pt`：

```bash
mkdir -p models/base
cp /path/to/mace-mp-0b3-medium.model models/base/

python -m mace.cli.create_lammps_model \
  models/base/mace-mp-0b3-medium.model \
  --format=mliap
```

转换后会得到 `models/base/mace-mp-0b3-medium.model-mliap_lammps.pt`。

### 3. 生成并提交基础模型 NPT

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

`--temperature` 和 `--pressure` 都支持多个值，会按笛卡尔积生成条件目录，例如：

```text
outputs/run_mace_md_npt_all.sh
outputs/001_LiCl/973K_00bar/mace-md-npt/run.sh
outputs/001_LiCl/1073K_00bar/mace-md-npt/run.sh
```

运行脚本默认使用 `/root/base/envs/mace-lammps-torch291-cu128` 和 `/root/base/software/lammps-mliap-torch291-cu128`。如果你的安装路径不同，在提交前设置：

```bash
export MACE_ENV=/path/to/mace-lammps-torch291-cu128
export LAMMPS_PREFIX=/path/to/lammps-mliap-install
```

任务成功结束后会写入空文件 `task_finished`。如果任务中断，重新执行总脚本时，已有 `task_finished` 的目录会自动跳过。

### 4. 生成并提交基础模型 NVT

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

`mace-md-nvt` 会跟随已有的 NPT 条件目录生成任务。默认 NVT 读取 `simulation_box/box.data`，因此不依赖 NPT 是否已经完成。如果只想为部分温度生成 NVT，可加入 `--temperature 973`。如果希望 NVT 盒子使用 NPT 后段平均密度，可加入 `--use-npt-density 0.8`；此时对应 NPT 目录需要已经生成 `final_mace_npt.data` 和 `log.kk.half.lammps`。

### 5. 准备并运行 MACE 微调

训练数据使用 extxyz 文件。下面示例假设训练集和验证集分别为 `data/finetune/Training.extxyz` 和 `data/finetune/Validation.extxyz`：

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

该命令会在 `outputs/finetune-mace/` 下创建：

```text
outputs/finetune-mace/
├── checkpoints/
├── logs/
├── models/
├── results/
├── finetune_settings.json
└── run_train.sh
```

`--smoke-max-configs` 是可选参数，只用于 smoke 测试。正式训练时可以删除该参数。

### 6. 使用微调模型生成 MD 任务

第 5 步提交训练任务时已经进入 `outputs/finetune-mace` 目录。训练完成后，先回到项目根目录：

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

`--task-suffix finetuned` 会生成 `mace-md-npt-finetuned` 和 `mace-md-nvt-finetuned`，避免覆盖基础模型任务。微调模型通常只支持微调数据中出现过的元素；如果某个体系包含训练数据未覆盖的元素，LAMMPS 可能在 `pair_coeff` 阶段报错。

## 参考资料

- [Python venv](https://docs.python.org/3.12/tutorial/venv.html)
- [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
- [MACE PyPI](https://pypi.org/project/mace-torch/)
- [NVIDIA cuEquivariance](https://docs.nvidia.com/cuda/cuequivariance/)
- [CuPy](https://docs.cupy.dev/en/v13.6.0/install.html)
- [MACE ML-IAP](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)
- [LAMMPS Python](https://docs.lammps.org/Python_install.html)
- [Packmol conda-forge](https://anaconda.org/conda-forge/packmol)
- [Open MPI conda-forge](https://anaconda.org/conda-forge/openmpi)
