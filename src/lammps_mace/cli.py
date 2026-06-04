from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__
from .build_boxes import build_boxes_from_csv
from .config import write_example_systems_csv
from .md import (
    DEFAULT_MODEL_PATH,
    aggregate_run_script_name,
    render_npt_inputs_for_boxes,
    render_nvt_inputs_for_boxes,
)
from .train import (
    DEFAULT_FINETUNE_DATASET_ROOT,
    DEFAULT_FINETUNE_BATCH_SIZE,
    DEFAULT_FINETUNE_MAX_NUM_EPOCHS,
    DEFAULT_FINETUNE_TASK_DIR,
    DEFAULT_FINETUNE_NAME,
    DEFAULT_FINETUNE_TRAIN_FILE,
    DEFAULT_FINETUNE_VALID_FILE,
    prepare_finetune_task,
)


DEFAULT_BOXES_ROOT = "outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lammps-mace")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="write a starter CSV into a target directory")
    init_parser.add_argument("--target", required=True, help="target project directory")
    init_parser.set_defaults(func=_cmd_init)

    boxes_parser = subparsers.add_parser("build-boxes", help="build one system folder per CSV row")
    boxes_parser.add_argument("--csv", required=True, help="high-throughput systems CSV")
    boxes_parser.add_argument(
        "--output-root",
        default=DEFAULT_BOXES_ROOT,
        help=f"directory that will contain per-system folders (default: {DEFAULT_BOXES_ROOT})",
    )
    boxes_parser.add_argument(
        "--density",
        dest="density_g_cm3",
        type=float,
        required=True,
        help="box-building density in g/cm^3",
    )
    boxes_parser.add_argument(
        "--packmol-tolerance",
        dest="packmol_tolerance_angstrom",
        type=float,
        default=2.0,
        help="minimum inter-atom distance passed to Packmol (default: 2.0)",
    )
    boxes_parser.add_argument(
        "--packmol-bin",
        default="packmol",
        help="Packmol executable name or path (default: packmol)",
    )
    boxes_parser.add_argument("--seed", type=int, default=973, help="Packmol random seed (default: 973)")
    boxes_parser.set_defaults(func=_cmd_build_boxes)

    npt_parser = subparsers.add_parser(
        "mace-md-npt",
        help="render NPT MACE-MD inputs under each built system folder",
    )
    _add_stage_md_arguments(npt_parser)
    _add_npt_runtime_arguments(npt_parser)
    npt_parser.add_argument("--temperature", type=float, nargs="+", required=True, help="NPT temperature(s) in K")
    npt_parser.add_argument("--pressure", type=float, nargs="+", required=True, help="NPT pressure(s) in bar")
    npt_parser.set_defaults(func=_cmd_mace_md_npt)

    nvt_parser = subparsers.add_parser(
        "mace-md-nvt",
        help="render NVT MACE-MD inputs from completed NPT results",
    )
    _add_stage_md_arguments(nvt_parser)
    _add_nvt_runtime_arguments(nvt_parser)
    nvt_parser.add_argument("--temperature", type=float, nargs="+", help="only render NVT for these NPT temperature(s)")
    nvt_parser.add_argument(
        "--use-npt-density",
        type=float,
        help="use the final fraction of NPT density samples to size the NVT box, for example 0.8",
    )
    nvt_parser.set_defaults(func=_cmd_mace_md_nvt)

    finetune_parser = subparsers.add_parser("finetune", help="prepare a MACE fine-tuning task folder")
    finetune_parser.add_argument(
        "--task-dir",
        default=DEFAULT_FINETUNE_TASK_DIR,
        help=f"fine-tuning task directory (default: {DEFAULT_FINETUNE_TASK_DIR})",
    )
    finetune_parser.add_argument(
        "--dataset-root",
        default=DEFAULT_FINETUNE_DATASET_ROOT,
        help=f"directory containing fine-tuning extxyz files (default: {DEFAULT_FINETUNE_DATASET_ROOT})",
    )
    _add_train_arguments(
        finetune_parser,
        train_file_default=DEFAULT_FINETUNE_TRAIN_FILE,
        valid_file_default=DEFAULT_FINETUNE_VALID_FILE,
        name_default=DEFAULT_FINETUNE_NAME,
    )
    finetune_parser.add_argument(
        "--smoke-max-configs",
        type=int,
        help="write smoke-sized train/valid extxyz files with at most this many configs each",
    )
    finetune_parser.add_argument(
        "--require-dataset",
        action="store_true",
        help="fail if the selected train/valid extxyz files are missing",
    )
    finetune_parser.set_defaults(func=_cmd_finetune)

    return parser


def _add_stage_md_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--boxes-root",
        default=DEFAULT_BOXES_ROOT,
        help=f"directory containing per-system folders (default: {DEFAULT_BOXES_ROOT})",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"ML-IAP model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--task-suffix",
        help=(
            "append a suffix to generated task folders and aggregate scripts, "
            "for example finetuned -> mace-md-npt-finetuned"
        ),
    )


def _add_shared_md_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timestep-fs", type=float, default=1.0, help="LAMMPS timestep in fs (default: 1.0)")
    parser.add_argument("--thermo-every", type=int, default=100, help="thermo output interval (default: 100)")
    parser.add_argument("--dump-every", type=int, default=100, help="trajectory dump interval (default: 100)")
    parser.add_argument("--tdamp-fs", type=float, default=100.0, help="thermostat damping in fs (default: 100.0)")


def _add_npt_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    _add_shared_md_runtime_arguments(parser)
    parser.add_argument(
        "--npt-steps",
        type=int,
        default=1000,
        help="number of MD steps for generated NPT inputs (default: 1000)",
    )
    parser.add_argument("--pdamp-fs", type=float, default=1000.0, help="barostat damping in fs (default: 1000.0)")


def _add_nvt_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    _add_shared_md_runtime_arguments(parser)
    parser.add_argument(
        "--nvt-steps",
        type=int,
        default=1000,
        help="number of MD steps for generated NVT inputs (default: 1000)",
    )


def _add_train_arguments(
    parser: argparse.ArgumentParser,
    *,
    train_file_default: str = "data/dft/train.extxyz",
    valid_file_default: str = "data/dft/valid.extxyz",
    name_default: str = DEFAULT_FINETUNE_NAME,
) -> None:
    parser.add_argument("--foundation-model", default="models/base/mace-mp-0b3-medium.model")
    parser.add_argument("--train-file", default=train_file_default)
    parser.add_argument("--valid-file", default=valid_file_default)
    parser.add_argument("--test-file")
    parser.add_argument("--name", default=name_default)
    parser.add_argument("--results-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--log-dir")
    parser.add_argument("--model-dir")
    parser.add_argument("--checkpoints-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-num-epochs", type=int, default=DEFAULT_FINETUNE_MAX_NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_FINETUNE_BATCH_SIZE)
    parser.add_argument("--valid-batch-size", type=int)
    parser.add_argument("--energy-key", default="energy")
    parser.add_argument("--forces-key", default="forces")
    parser.add_argument("--stress-key")
    parser.add_argument("--e0s", dest="E0s", default="average")
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--forces-weight", type=float, default=1.0)
    parser.add_argument("--stress-weight", type=float)
    parser.add_argument("--r-max", type=float)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--scaling", default="rms_forces_scaling")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--amsgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-dtype", default="float64")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--multiheads-finetuning", action=argparse.BooleanOptionalAction, default=False)


def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.target)
    csv_path = target / "examples" / "systems.csv"
    write_example_systems_csv(csv_path)
    for directory in (DEFAULT_BOXES_ROOT, "models/base", "models/finetuned", "data/finetune", "data/dft"):
        (target / directory).mkdir(parents=True, exist_ok=True)
    print(f"wrote {csv_path}")
    return 0


def _cmd_build_boxes(args: argparse.Namespace) -> int:
    results = build_boxes_from_csv(
        args.csv,
        args.output_root,
        density_g_cm3=args.density_g_cm3,
        packmol_tolerance_angstrom=args.packmol_tolerance_angstrom,
        packmol_executable=args.packmol_bin,
        seed=args.seed,
    )
    for result in results:
        print(
            f"{result.system_index}: {result.atom_count} atoms, "
            f"L={result.box_length_angstrom:.6f} A -> {result.data_file}"
        )
    print(f"built {len(results)} system folders under {args.output_root}")
    return 0


def _cmd_mace_md_npt(args: argparse.Namespace) -> int:
    results = render_npt_inputs_for_boxes(
        _md_settings_from_args(args),
        args.boxes_root,
        temperatures_k=args.temperature,
        pressures_bar=args.pressure,
        model_path=args.model_path,
        task_suffix=args.task_suffix,
    )
    for result in results:
        print(f"{result.system_index}: {result.ensemble.upper()} input -> {result.input_file}")
    print(f"rendered {len(results)} NPT LAMMPS input files under {args.boxes_root}")
    print(f"run script -> {Path(args.boxes_root) / aggregate_run_script_name('npt', args.task_suffix)}")
    return 0


def _cmd_mace_md_nvt(args: argparse.Namespace) -> int:
    results = render_nvt_inputs_for_boxes(
        _md_settings_from_args(args),
        args.boxes_root,
        temperatures_k=args.temperature,
        model_path=args.model_path,
        use_npt_density_fraction=args.use_npt_density,
        task_suffix=args.task_suffix,
    )
    for result in results:
        print(f"{result.system_index}: {result.ensemble.upper()} input -> {result.input_file}")
    print(f"rendered {len(results)} NVT LAMMPS input files under {args.boxes_root}")
    print(f"run script -> {Path(args.boxes_root) / aggregate_run_script_name('nvt', args.task_suffix)}")
    return 0


def _cmd_finetune(args: argparse.Namespace) -> int:
    result = prepare_finetune_task(
        task_dir=args.task_dir,
        dataset_root=args.dataset_root,
        train_file=args.train_file,
        valid_file=args.valid_file,
        foundation_model=args.foundation_model,
        name=args.name,
        device=args.device,
        max_num_epochs=args.max_num_epochs,
        batch_size=args.batch_size,
        smoke_max_configs=args.smoke_max_configs,
        require_dataset=args.require_dataset,
        extra_train_settings=_optional_train_settings_from_args(args),
    )
    print(f"prepared MACE fine-tuning task -> {result.task_dir}")
    print(f"settings -> {result.settings_file}")
    print(f"run script -> {result.run_script}")
    if result.missing_dataset_files:
        missing = ", ".join(str(path) for path in result.missing_dataset_files)
        print(f"warning: missing expected dataset files: {missing}")
    return 0


def _md_settings_from_args(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "npt_steps": getattr(args, "npt_steps", 1000),
        "nvt_steps": getattr(args, "nvt_steps", 1000),
        "timestep_fs": args.timestep_fs,
        "thermo_every": args.thermo_every,
        "dump_every": args.dump_every,
        "tdamp_fs": args.tdamp_fs,
        "pdamp_fs": getattr(args, "pdamp_fs", 1000.0),
    }


def _optional_train_settings_from_args(args: argparse.Namespace) -> dict[str, object]:
    settings: dict[str, object] = {}
    for key in (
        "test_file",
        "results_dir",
        "work_dir",
        "log_dir",
        "model_dir",
        "checkpoints_dir",
        "energy_key",
        "forces_key",
        "stress_key",
        "E0s",
        "valid_batch_size",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "r_max",
        "lr",
        "scaling",
        "ema",
        "ema_decay",
        "amsgrad",
        "default_dtype",
        "seed",
        "multiheads_finetuning",
    ):
        value = getattr(args, key)
        default = {
            "test_file": None,
            "results_dir": None,
            "work_dir": None,
            "log_dir": None,
            "model_dir": None,
            "checkpoints_dir": None,
            "energy_key": "energy",
            "forces_key": "forces",
            "stress_key": None,
            "E0s": "average",
            "valid_batch_size": None,
            "energy_weight": 1.0,
            "forces_weight": 1.0,
            "stress_weight": None,
            "r_max": None,
            "lr": 0.01,
            "scaling": "rms_forces_scaling",
            "ema": True,
            "ema_decay": 0.99,
            "amsgrad": True,
            "default_dtype": "float64",
            "seed": 3,
            "multiheads_finetuning": False,
        }[key]
        if value != default:
            settings[key] = value
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
