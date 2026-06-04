from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import shlex
import sys
from typing import Any


DEFAULT_FINETUNE_DATASET_ROOT = "data/finetune"
DEFAULT_FINETUNE_TASK_DIR = "outputs/finetune-mace"
DEFAULT_FINETUNE_NAME = "mace-mp-0b3-medium-finetuned"
DEFAULT_FINETUNE_TRAIN_FILE = "train.extxyz"
DEFAULT_FINETUNE_VALID_FILE = "valid.extxyz"
DEFAULT_FINETUNE_MAX_NUM_EPOCHS = 6
DEFAULT_FINETUNE_BATCH_SIZE = 2


@dataclass(frozen=True)
class FinetuneTaskResult:
    task_dir: Path
    settings_file: Path
    run_script: Path
    command: list[str]
    missing_dataset_files: tuple[Path, ...]
    smoke_dataset_dir: Path | None = None


def build_train_command(config: dict[str, Any]) -> list[str]:
    train = config.get("train", {})
    command = [
        "mace_run_train",
        "--name",
        str(train.get("name", DEFAULT_FINETUNE_NAME)),
        "--foundation_model",
        str(train.get("foundation_model", "models/base/mace-mp-0b3-medium.model")),
        "--train_file",
        str(train.get("train_file", "data/dft/train.extxyz")),
        "--valid_file",
        str(train.get("valid_file", "data/dft/valid.extxyz")),
        "--results_dir",
        str(train.get("results_dir", train.get("output_dir", "models/finetuned"))),
    ]

    for key, option in (
        ("work_dir", "--work_dir"),
        ("log_dir", "--log_dir"),
        ("model_dir", "--model_dir"),
        ("checkpoints_dir", "--checkpoints_dir"),
        ("test_file", "--test_file"),
        ("E0s", "--E0s"),
        ("valid_batch_size", "--valid_batch_size"),
    ):
        if train.get(key):
            command.extend([option, str(train[key])])

    command.extend(
        [
            "--device",
            str(train.get("device", "cuda")),
            "--max_num_epochs",
            str(train.get("max_num_epochs", 50)),
            "--batch_size",
            str(train.get("batch_size", 4)),
        ]
    )

    for key in (
        "energy_key",
        "forces_key",
        "stress_key",
        "energy_weight",
        "forces_weight",
        "stress_weight",
        "r_max",
        "lr",
        "scaling",
        "default_dtype",
        "seed",
        "multiheads_finetuning",
    ):
        if train.get(key) is not None:
            command.extend([f"--{key}", str(train[key])])

    if train.get("ema"):
        command.append("--ema")
    if train.get("ema_decay") is not None:
        command.extend(["--ema_decay", str(train["ema_decay"])])
    if train.get("amsgrad"):
        command.append("--amsgrad")

    return command


def shell_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def prepare_finetune_task(
    *,
    task_dir: str | Path = DEFAULT_FINETUNE_TASK_DIR,
    dataset_root: str | Path = DEFAULT_FINETUNE_DATASET_ROOT,
    train_file: str | Path = DEFAULT_FINETUNE_TRAIN_FILE,
    valid_file: str | Path = DEFAULT_FINETUNE_VALID_FILE,
    foundation_model: str | Path = "models/base/mace-mp-0b3-medium.model",
    name: str = DEFAULT_FINETUNE_NAME,
    device: str = "cuda",
    max_num_epochs: int = DEFAULT_FINETUNE_MAX_NUM_EPOCHS,
    batch_size: int = DEFAULT_FINETUNE_BATCH_SIZE,
    smoke_max_configs: int | None = None,
    require_dataset: bool = False,
    extra_train_settings: dict[str, Any] | None = None,
) -> FinetuneTaskResult:
    """Create a reproducible MACE fine-tuning task folder from extxyz data."""
    root = Path(task_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("logs", "models", "checkpoints", "results"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    dataset = Path(dataset_root).expanduser()
    source_train_file = _resolve_dataset_file(dataset, train_file)
    source_valid_file = _resolve_dataset_file(dataset, valid_file)
    expected_files = (source_train_file, source_valid_file)
    missing = tuple(path for path in expected_files if not path.exists())
    if require_dataset and missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing fine-tuning dataset files: {missing_text}")

    selected_train_file = source_train_file
    selected_valid_file = source_valid_file
    smoke_dataset_dir: Path | None = None
    if smoke_max_configs is not None:
        if smoke_max_configs <= 0:
            raise ValueError("smoke_max_configs must be positive.")
        missing_smoke_sources = tuple(path for path in (source_train_file, source_valid_file) if not path.exists())
        if missing_smoke_sources:
            missing_text = ", ".join(str(path) for path in missing_smoke_sources)
            raise FileNotFoundError(f"Missing files required for smoke dataset: {missing_text}")
        smoke_dataset_dir = root / "smoke-dataset"
        smoke_dataset_dir.mkdir(parents=True, exist_ok=True)
        selected_train_file = smoke_dataset_dir / Path(train_file).name
        selected_valid_file = smoke_dataset_dir / Path(valid_file).name
        _write_smoke_extxyz(source_train_file, selected_train_file, smoke_max_configs)
        _write_smoke_extxyz(source_valid_file, selected_valid_file, smoke_max_configs)

    task_abs = root.resolve()
    settings: dict[str, Any] = {
        "name": name,
        "foundation_model": str(Path(foundation_model).expanduser().resolve()),
        "train_file": str(selected_train_file.resolve()),
        "valid_file": str(selected_valid_file.resolve()),
        "results_dir": str((task_abs / "results").resolve()),
        "work_dir": str(task_abs),
        "log_dir": str((task_abs / "logs").resolve()),
        "model_dir": str((task_abs / "models").resolve()),
        "checkpoints_dir": str((task_abs / "checkpoints").resolve()),
        "device": device,
        "max_num_epochs": max_num_epochs,
        "batch_size": batch_size,
        "valid_batch_size": batch_size,
        "multiheads_finetuning": False,
        "E0s": "average",
        "energy_key": "energy",
        "forces_key": "forces",
        "energy_weight": 1.0,
        "forces_weight": 1.0,
        "lr": 0.01,
        "scaling": "rms_forces_scaling",
        "ema": True,
        "ema_decay": 0.99,
        "amsgrad": True,
        "default_dtype": "float64",
        "seed": 3,
    }
    settings.update(extra_train_settings or {})
    if settings.get("test_file"):
        settings["test_file"] = str(_resolve_dataset_file(dataset, str(settings["test_file"])).resolve())
    if settings.get("output_dir"):
        settings["model_dir"] = str(Path(settings["output_dir"]).expanduser().resolve())
    for directory_key in ("results_dir", "work_dir", "log_dir", "model_dir", "checkpoints_dir"):
        settings[directory_key] = str(Path(settings[directory_key]).expanduser().resolve())
        Path(settings[directory_key]).mkdir(parents=True, exist_ok=True)

    settings_file = root / "finetune_settings.json"
    run_script = root / "run_train.sh"
    command = build_train_command({"train": settings})
    command[0] = _resolve_mace_run_train_executable()
    dataset_settings = _compact_json_mapping(
        {
            "root": str(dataset.resolve()),
            "source_train_file": str(source_train_file.resolve()),
            "source_valid_file": str(source_valid_file.resolve()),
            "smoke_max_configs": smoke_max_configs,
            "smoke_dataset_dir": str(smoke_dataset_dir.resolve()) if smoke_dataset_dir else None,
            "missing_files": [str(path.resolve()) for path in missing],
        }
    )
    settings_file.write_text(
        json.dumps(
            {
                "stage": "finetune",
                "task_dir": str(task_abs),
                "dataset": dataset_settings,
                "run_script": run_script.name,
                "command": shell_command(command),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")\"\n"
        f"exec {shell_command(command)} \"$@\"\n",
        encoding="utf-8",
    )
    os.chmod(run_script, 0o755)

    return FinetuneTaskResult(root, settings_file, run_script, command, missing, smoke_dataset_dir)


def _compact_json_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value not in (None, [], {})}


def _resolve_dataset_file(dataset_root: Path, file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    return dataset_root / path


def _write_smoke_extxyz(source: Path, output: Path, max_configs: int) -> int:
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        while written < max_configs:
            atom_count_line = src.readline()
            if not atom_count_line:
                break
            if not atom_count_line.strip():
                continue
            try:
                atom_count = int(atom_count_line.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid extxyz frame atom count in {source}: {atom_count_line.strip()}") from exc
            comment_line = src.readline()
            if not comment_line:
                raise ValueError(f"Incomplete extxyz frame in {source}: missing comment line")
            atom_lines = [src.readline() for _ in range(atom_count)]
            if any(line == "" for line in atom_lines):
                raise ValueError(f"Incomplete extxyz frame in {source}: missing atom lines")
            dst.write(atom_count_line)
            dst.write(comment_line)
            dst.writelines(atom_lines)
            written += 1
    return written


def _resolve_mace_run_train_executable() -> str:
    venv_executable = Path(sys.executable).with_name("mace_run_train")
    if venv_executable.exists():
        return str(venv_executable)
    return shutil.which("mace_run_train") or "mace_run_train"
