from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .box import AMU_TO_GRAM, CM3_TO_ANGSTROM3
from .chemistry import DEFAULT_MASSES


DEFAULT_MODEL_PATH = "models/base/mace-mp-0b3-medium.model-mliap_lammps.pt"
SIMULATION_BOX_DIR = "simulation_box"
NPT_TASK_DIR = "mace-md-npt"
NVT_TASK_DIR = "mace-md-nvt"
STAGE_RUN_SCRIPT = "run.sh"
NPT_RUN_LABEL = "mace_npt"
NVT_RUN_LABEL = "mace_nvt"

DEFAULT_MD_SETTINGS: dict[str, float | int] = {
    "timestep_fs": 1.0,
    "npt_steps": 1000,
    "nvt_steps": 1000,
    "thermo_every": 100,
    "dump_every": 100,
    "tdamp_fs": 100.0,
    "pdamp_fs": 1000.0,
}


@dataclass(frozen=True)
class MdInputResult:
    system_index: str
    ensemble: str
    condition: str
    folder: Path
    input_file: Path
    run_script: Path


@dataclass(frozen=True)
class NptDensityResult:
    source_log: Path
    average_density_g_cm3: float
    box_length_angstrom: float
    sample_count: int
    used_sample_count: int
    used_fraction: float


@dataclass(frozen=True)
class Condition:
    temperature_k: float
    pressure_bar: float
    folder_name: str


def render_npt_inputs_for_boxes(
    md_settings: dict[str, Any] | None,
    boxes_root: str | Path,
    *,
    temperatures_k: list[float],
    pressures_bar: list[float],
    model_path: str | Path | None = None,
    task_suffix: str | None = None,
) -> list[MdInputResult]:
    if not temperatures_k:
        raise ValueError("At least one NPT temperature must be provided.")
    if not pressures_bar:
        raise ValueError("At least one NPT pressure must be provided.")

    root = _boxes_root(boxes_root)
    box_folders = _box_folders(root)
    settings = {**DEFAULT_MD_SETTINGS, **(md_settings or {})}
    selected_model_path = resolve_model_path(model_path)
    npt_task_dir = task_dir_name("npt", task_suffix)
    results: list[MdInputResult] = []

    for box_folder in box_folders:
        system = _system_metadata(box_folder)
        for temperature_k in temperatures_k:
            for pressure_bar in pressures_bar:
                condition = condition_from_values(temperature_k, pressure_bar)
                condition_root = box_folder / condition.folder_name
                output_dir = condition_root / npt_task_dir
                input_file = output_dir / "in.lammps"
                output_dir.mkdir(parents=True, exist_ok=True)
                input_data = os.path.relpath(box_folder / SIMULATION_BOX_DIR / "box.data", output_dir)
                input_file.write_text(
                    render_lammps_input_text(
                        settings,
                        ensemble="npt",
                        input_data=input_data,
                        species_order=_species_order(system),
                        temperature_k=condition.temperature_k,
                        pressure_bar=condition.pressure_bar,
                        model_path=selected_model_path,
                        run_label=NPT_RUN_LABEL,
                    ),
                    encoding="utf-8",
                )
                run_script = _write_ensemble_run_script(output_dir)
                _write_stage_settings(
                    output_dir / "md_settings.json",
                    selected_model_path,
                    settings,
                    box_folder / SIMULATION_BOX_DIR / "system.json",
                    condition,
                )
                results.append(
                    MdInputResult(
                        str(system.get("index", box_folder.name)),
                        "npt",
                        condition.folder_name,
                        output_dir,
                        input_file,
                        run_script,
                    )
                )

    _write_aggregate_run_script(root, [result.run_script for result in results], "npt", task_suffix=task_suffix)
    return results


def render_nvt_inputs_for_boxes(
    md_settings: dict[str, Any] | None,
    boxes_root: str | Path,
    *,
    temperatures_k: list[float] | None = None,
    model_path: str | Path | None = None,
    use_npt_density_fraction: float | None = None,
    task_suffix: str | None = None,
) -> list[MdInputResult]:
    root = _boxes_root(boxes_root)
    box_folders = _box_folders(root)
    settings = {**DEFAULT_MD_SETTINGS, **(md_settings or {})}
    selected_model_path = resolve_model_path(model_path)
    requested_temperatures = None if temperatures_k is None else {condition_temperature_token(value) for value in temperatures_k}
    npt_task_dir = task_dir_name("npt", task_suffix)
    nvt_task_dir = task_dir_name("nvt", task_suffix)
    results: list[MdInputResult] = []

    for box_folder in box_folders:
        system = _system_metadata(box_folder)
        for condition_root, condition in _npt_condition_roots(box_folder, npt_task_dir):
            if requested_temperatures is not None and condition_temperature_token(condition.temperature_k) not in requested_temperatures:
                continue

            output_dir = condition_root / nvt_task_dir
            input_file = output_dir / "in.lammps"
            output_dir.mkdir(parents=True, exist_ok=True)
            input_source = box_folder / SIMULATION_BOX_DIR / "box.data"
            density_result: NptDensityResult | None = None

            if use_npt_density_fraction is not None:
                npt_dir = condition_root / npt_task_dir
                npt_data = npt_dir / f"final_{NPT_RUN_LABEL}.data"
                npt_log = npt_dir / "log.kk.half.lammps"
                if not npt_data.exists():
                    raise ValueError(f"Missing NPT final data: {npt_data}. Run this NPT task first.")
                density_result = average_npt_density(
                    npt_log,
                    use_npt_density_fraction,
                    _element_counts(system),
                )
                input_source = npt_data

            input_data = os.path.relpath(input_source, output_dir)
            input_file.write_text(
                render_lammps_input_text(
                    settings,
                    ensemble="nvt",
                    input_data=input_data,
                    species_order=_species_order(system),
                    temperature_k=condition.temperature_k,
                    pressure_bar=condition.pressure_bar,
                    model_path=selected_model_path,
                    run_label=NVT_RUN_LABEL,
                    box_length_angstrom=None if density_result is None else density_result.box_length_angstrom,
                    density_note=None
                    if density_result is None
                    else (
                        f"NVT box length from average NPT density "
                        f"{density_result.average_density_g_cm3:.8f} g/cm^3 "
                        f"using last {density_result.used_fraction:.3g} of samples "
                        f"({density_result.used_sample_count}/{density_result.sample_count})"
                    ),
                ),
                encoding="utf-8",
            )
            if density_result is not None:
                _write_nvt_density_metadata(output_dir / "nvt_density.json", density_result, input_source)
            run_script = _write_ensemble_run_script(output_dir)
            _write_stage_settings(
                output_dir / "md_settings.json",
                selected_model_path,
                settings,
                box_folder / SIMULATION_BOX_DIR / "system.json",
                condition,
            )
            results.append(
                MdInputResult(
                    str(system.get("index", box_folder.name)),
                    "nvt",
                    condition.folder_name,
                    output_dir,
                    input_file,
                    run_script,
                )
            )

    if not results:
        raise ValueError("No matching NPT task folders were found for NVT input generation.")

    _write_aggregate_run_script(root, [result.run_script for result in results], "nvt", task_suffix=task_suffix)
    return results


def resolve_model_path(
    override: str | Path | None = None,
) -> str:
    raw_path: str | Path = DEFAULT_MODEL_PATH if override is None else override
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def average_npt_density(
    log_path: str | Path,
    used_fraction: float,
    element_counts: dict[str, int],
) -> NptDensityResult:
    fraction = float(used_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--use-npt-density must be between 0 and 1.")

    source_log = Path(log_path)
    densities = _read_lammps_thermo_column(source_log, "Density")
    if not densities:
        raise ValueError(f"No Density samples were found in NPT log: {source_log}")
    used_count = max(1, math.ceil(len(densities) * fraction))
    selected = densities[-used_count:]
    average_density = sum(selected) / len(selected)
    box_length = _box_length_from_density(element_counts, average_density)
    return NptDensityResult(
        source_log=source_log,
        average_density_g_cm3=average_density,
        box_length_angstrom=box_length,
        sample_count=len(densities),
        used_sample_count=len(selected),
        used_fraction=fraction,
    )


def render_lammps_input_text(
    md_settings: dict[str, Any] | None,
    *,
    ensemble: str,
    input_data: str | Path,
    species_order: list[str],
    temperature_k: float,
    pressure_bar: float,
    model_path: str | Path,
    run_label: str,
    box_length_angstrom: float | None = None,
    density_note: str | None = None,
) -> str:
    settings = {**DEFAULT_MD_SETTINGS, **(md_settings or {})}
    selected_ensemble = str(ensemble).lower()
    if selected_ensemble not in {"npt", "nvt"}:
        raise ValueError(f"Unsupported ensemble: {selected_ensemble}")

    selected_input_data = str(input_data)
    selected_model_path = str(model_path)
    temperature = float(temperature_k)
    timestep_fs = float(settings["timestep_fs"])
    thermo_every = int(settings["thermo_every"])
    dump_every = int(settings["dump_every"])
    steps = int(settings[f"{selected_ensemble}_steps"])
    tdamp = float(settings["tdamp_fs"]) / 1000.0
    species = " ".join(species_order)

    fix_line = f"fix integrator all nvt temp {temperature:.1f} {temperature:.1f} {tdamp:.4f}"
    if selected_ensemble == "npt":
        pdamp = float(settings["pdamp_fs"]) / 1000.0
        pressure = float(pressure_bar)
        fix_line = (
            f"fix integrator all npt temp {temperature:.1f} {temperature:.1f} {tdamp:.4f} "
            f"iso {pressure:.4f} {pressure:.4f} {pdamp:.4f}"
        )

    box_resize = ""
    if box_length_angstrom is not None:
        length = float(box_length_angstrom)
        note = f"# {density_note}\n" if density_note else ""
        box_resize = (
            "\n"
            f"{note}"
            f"change_box all x final 0.0 {length:.6f} "
            f"y final 0.0 {length:.6f} "
            f"z final 0.0 {length:.6f} remap units box\n"
        )

    return f"""# Generated by lammps-mace for {selected_ensemble.upper()}
units metal
atom_style atomic
boundary p p p
newton on

read_data {selected_input_data}
{box_resize}
pair_style mliap unified {selected_model_path} 0
pair_coeff * * {species}

neighbor 2.0 bin
neigh_modify delay 0 every 1 check yes

timestep {timestep_fs / 1000.0:.6f}
thermo {thermo_every}
thermo_style custom step temp pe ke etotal press density vol

velocity all create {temperature:.1f} 973 mom yes rot yes dist gaussian

compute rdf_all all rdf 200
fix rdf_out all ave/time {thermo_every} 1 {thermo_every} c_rdf_all[*] file rdf_{run_label}.dat mode vector
compute msd_all all msd
fix msd_out all ave/time {thermo_every} 1 {thermo_every} c_msd_all[*] file msd_{run_label}.dat

dump traj all custom {dump_every} traj_{run_label}.lammpstrj id type x y z
dump_modify traj sort id

{fix_line}
run {steps}

write_data final_{run_label}.data
write_restart final_{run_label}.restart
"""


def aggregate_run_script_name(ensemble: str, task_suffix: str | None = None) -> str:
    suffix = normalize_task_suffix(task_suffix)
    suffix_part = f"_{suffix}" if suffix else ""
    return f"run_mace_md_{ensemble}{suffix_part}_all.sh"


def task_dir_name(ensemble: str, task_suffix: str | None = None) -> str:
    selected_ensemble = str(ensemble).lower()
    if selected_ensemble == "npt":
        base = NPT_TASK_DIR
    elif selected_ensemble == "nvt":
        base = NVT_TASK_DIR
    else:
        raise ValueError(f"Unsupported ensemble: {selected_ensemble}")
    suffix = normalize_task_suffix(task_suffix)
    return f"{base}-{suffix}" if suffix else base


def normalize_task_suffix(task_suffix: str | None) -> str:
    if task_suffix is None:
        return ""
    suffix = str(task_suffix).strip()
    if not suffix:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", suffix):
        raise ValueError("--task-suffix must contain only letters, numbers, dots, underscores, or hyphens.")
    return suffix


def condition_from_values(temperature_k: float, pressure_bar: float) -> Condition:
    return Condition(
        temperature_k=float(temperature_k),
        pressure_bar=float(pressure_bar),
        folder_name=f"{condition_temperature_token(temperature_k)}_{condition_pressure_token(pressure_bar)}",
    )


def condition_temperature_token(temperature_k: float) -> str:
    return f"{_number_token(temperature_k)}K"


def condition_pressure_token(pressure_bar: float) -> str:
    return f"{_number_token(pressure_bar, min_integer_digits=2)}bar"


def _number_token(value: float, *, min_integer_digits: int = 0) -> str:
    number = float(value)
    if number.is_integer():
        raw = f"{abs(int(number)):0{min_integer_digits}d}"
    else:
        raw = f"{abs(number):g}".replace(".", "p")
    return f"m{raw}" if number < 0 else raw


def _parse_number_token(token: str) -> float:
    sign = -1.0 if token.startswith("m") else 1.0
    raw = token[1:] if token.startswith("m") else token
    return sign * float(raw.replace("p", "."))


def _condition_from_folder_name(folder_name: str) -> Condition | None:
    match = re.fullmatch(r"(?P<temperature>.+K)_(?P<pressure>.+bar)", folder_name)
    if not match:
        return None
    temperature = _parse_number_token(match.group("temperature").removesuffix("K"))
    pressure = _parse_number_token(match.group("pressure").removesuffix("bar"))
    return Condition(temperature, pressure, folder_name)


def _boxes_root(boxes_root: str | Path) -> Path:
    root = Path(boxes_root)
    if not root.exists():
        raise ValueError(f"Boxes root does not exist: {root}")
    return root


def _box_folders(root: Path) -> list[Path]:
    folders = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / SIMULATION_BOX_DIR / "box.data").exists()
    )
    if not folders:
        raise ValueError(f"No system folders containing {SIMULATION_BOX_DIR}/box.data were found under {root}")
    return folders


def _npt_condition_roots(box_folder: Path, npt_task_dir: str = NPT_TASK_DIR) -> list[tuple[Path, Condition]]:
    roots: list[tuple[Path, Condition]] = []
    for path in sorted(child for child in box_folder.iterdir() if child.is_dir()):
        condition = _condition_from_folder_name(path.name)
        if condition is not None and (path / npt_task_dir / "in.lammps").exists():
            roots.append((path, condition))
    return roots


def _read_system_metadata(box_folder: Path) -> dict[str, Any]:
    metadata_file = box_folder / SIMULATION_BOX_DIR / "system.json"
    if not metadata_file.exists():
        raise ValueError(
            f"Missing generated system metadata: {metadata_file}. "
            "Build boxes from the CSV with `lammps-mace build-boxes` first."
        )
    loaded = json.loads(metadata_file.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Metadata must contain a mapping: {metadata_file}")
    return loaded


def _system_metadata(box_folder: Path) -> dict[str, Any]:
    metadata = _read_system_metadata(box_folder)
    system = metadata.get("system", {})
    if not isinstance(system, dict):
        raise ValueError(f"Generated system metadata is invalid: {box_folder / SIMULATION_BOX_DIR / 'system.json'}")
    return system


def _species_order(system: dict[str, Any]) -> list[str]:
    element_counts = _element_counts(system)
    return [str(element) for element in element_counts]


def _element_counts(system: dict[str, Any]) -> dict[str, int]:
    element_counts = system.get("element_counts", {})
    if not isinstance(element_counts, dict) or not element_counts:
        raise ValueError("Generated system metadata is missing element_counts.")
    return {str(element): int(count) for element, count in element_counts.items()}


def _read_lammps_thermo_column(log_path: Path, column: str) -> list[float]:
    if not log_path.exists():
        raise ValueError(f"Missing NPT log: {log_path}. Run the corresponding NPT task first.")

    values: list[float] = []
    column_index: int | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Step":
            column_index = parts.index(column) if column in parts else None
            continue
        if column_index is None or len(parts) <= column_index:
            continue
        try:
            float(parts[0])
            values.append(float(parts[column_index]))
        except ValueError:
            continue
    return values


def _box_length_from_density(element_counts: dict[str, int], density_g_cm3: float) -> float:
    total_mass_amu = 0.0
    for element, count in element_counts.items():
        if element not in DEFAULT_MASSES:
            raise KeyError(f"Missing atomic mass for element '{element}'.")
        total_mass_amu += int(count) * float(DEFAULT_MASSES[element])
    volume_a3 = total_mass_amu * AMU_TO_GRAM / float(density_g_cm3) * CM3_TO_ANGSTROM3
    return volume_a3 ** (1.0 / 3.0)


def _write_stage_settings(
    output: Path,
    model_path: str,
    settings: dict[str, Any],
    system_json: Path,
    condition: Condition,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "task": output.parent.name,
                "condition": {
                    "temperature_k": condition.temperature_k,
                    "pressure_bar": condition.pressure_bar,
                    "folder": condition.folder_name,
                },
                "model_path": model_path,
                "md": settings,
                "source_system_json": str(system_json),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _write_nvt_density_metadata(output: Path, density_result: NptDensityResult, source_data: Path) -> Path:
    output.write_text(
        json.dumps(
            {
                "source_log": str(density_result.source_log),
                "source_data": str(source_data),
                "use_npt_density_fraction": density_result.used_fraction,
                "sample_count": density_result.sample_count,
                "used_sample_count": density_result.used_sample_count,
                "average_density_g_cm3": density_result.average_density_g_cm3,
                "box_length_angstrom": density_result.box_length_angstrom,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def _write_ensemble_run_script(run_dir: Path) -> Path:
    script = run_dir / STAGE_RUN_SCRIPT
    script.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

export MACE_ENV="${MACE_ENV:-/root/base/envs/mace-lammps-torch291-cu128}"
export LAMMPS_PREFIX="${LAMMPS_PREFIX:-/root/base/software/lammps-mliap-torch291-cu128}"
export PATH="$LAMMPS_PREFIX/bin:$MACE_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$LAMMPS_PREFIX/lib:$MACE_ENV/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$MACE_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export OMPI_MCA_mtl="${OMPI_MCA_mtl:-^ofi}"
export OMPI_MCA_pml="${OMPI_MCA_pml:-ob1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f task_finished ]]; then
  echo "skip $SCRIPT_DIR: task_finished exists"
  exit 0
fi

rm -f task_finished
"$LAMMPS_PREFIX/bin/lmp" -k on g 1 -sf kk -pk kokkos neigh half -in in.lammps -log log.kk.half.lammps
grep "Loop time" log.kk.half.lammps
touch task_finished
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _write_aggregate_run_script(
    root: Path,
    run_scripts: list[Path],
    ensemble: str,
    *,
    task_suffix: str | None = None,
) -> Path:
    script = root / aggregate_run_script_name(ensemble, task_suffix)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "$SCRIPT_DIR"',
        "",
    ]
    for run_script in run_scripts:
        relative_script = os.path.relpath(run_script, root)
        lines.extend(
            [
                f'echo "==> {relative_script}"',
                f'bash "{relative_script}"',
                "",
            ]
        )
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(0o755)
    return script
