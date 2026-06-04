from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from math import floor
from pathlib import Path
import re
from typing import Any

from .box import BoxBuildResult, build_box_from_element_counts
from .chemistry import molar_mass, parse_formula


FIELD_ALIASES = {
    "index": {"体系索引", "system_index", "system id", "system_id", "index", "id"},
    "salts": {"盐1-盐2-...-盐n", "盐", "盐列表", "salts", "salt", "salt_formulas"},
    "concentrations": {
        "盐浓度1-盐浓度2-...-盐浓度n",
        "盐浓度1-盐浓度2-...-盐n",
        "盐浓度",
        "浓度",
        "concentrations",
        "concentration",
        "salt_concentrations",
    },
    "concentration_type": {
        "盐浓度类型（mol% or wt%）",
        "盐浓度类型(mol% or wt%)",
        "盐浓度类型",
        "浓度类型",
        "concentration_type",
        "conc_type",
    },
    "total_atoms": {"total_atoms", "目标原子数", "总原子数", "原子数", "target_atoms", "atoms"},
}

_SPLIT_RE = re.compile(r"\s*(?:-|,|;|，|；|、)\s*")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.+-]+")
SIMULATION_BOX_DIR = "simulation_box"


@dataclass(frozen=True)
class BuildBoxesResult:
    system_index: str
    folder: Path
    data_file: Path
    metadata_file: Path
    atom_count: int
    box_length_angstrom: float


def build_boxes_from_csv(
    csv_path: str | Path,
    output_root: str | Path,
    *,
    density_g_cm3: float,
    packmol_tolerance_angstrom: float = 2.0,
    packmol_executable: str = "packmol",
    seed: int = 973,
) -> list[BuildBoxesResult]:
    """Build one system folder per row in a high-throughput CSV."""
    source = Path(csv_path)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    results: list[BuildBoxesResult] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {source}")
        field_map = _map_fields(reader.fieldnames)
        for row_number, row in enumerate(reader, start=1):
            system = _system_from_row(
                row,
                field_map,
                row_number=row_number,
                density_g_cm3=density_g_cm3,
            )
            folder = root / _folder_name(system)
            folder.mkdir(parents=True, exist_ok=True)
            simulation_box = folder / SIMULATION_BOX_DIR
            data_file = simulation_box / "box.data"
            build_result = build_box_from_element_counts(
                system["element_counts"],
                system["density_g_cm3"],
                data_file,
                packmol_tolerance_angstrom=packmol_tolerance_angstrom,
                packmol_executable=packmol_executable,
                seed=seed + row_number - 1,
            )
            metadata_file = simulation_box / "system.json"
            _write_metadata(metadata_file, source, system, build_result)
            results.append(
                BuildBoxesResult(
                    system_index=system["index"],
                    folder=folder,
                    data_file=data_file,
                    metadata_file=metadata_file,
                    atom_count=build_result.atom_count,
                    box_length_angstrom=build_result.box_length_angstrom,
                )
            )
    return results


def composition_from_salts(
    salts: list[str],
    concentrations: list[float],
    concentration_type: str,
    total_atoms: int,
) -> tuple[dict[str, int], dict[str, int], list[float]]:
    if len(salts) != len(concentrations):
        raise ValueError("Salt and concentration counts do not match.")
    if total_atoms <= 0:
        raise ValueError("total_atoms must be positive.")

    formula_counts = [parse_formula(salt) for salt in salts]
    mole_fractions = _mole_fractions(formula_counts, concentrations, concentration_type)
    atoms_per_unit = [sum(counts.values()) for counts in formula_counts]
    avg_atoms_per_unit = sum(frac * atoms for frac, atoms in zip(mole_fractions, atoms_per_unit))
    formula_unit_total = max(1, round(total_atoms / avg_atoms_per_unit))
    unit_counts = _round_units(mole_fractions, formula_unit_total)

    element_counts: defaultdict[str, int] = defaultdict(int)
    for unit_count, counts in zip(unit_counts, formula_counts):
        for element, count in counts.items():
            element_counts[element] += unit_count * count

    return dict(element_counts), dict(zip(salts, unit_counts)), mole_fractions


def _system_from_row(
    row: dict[str, str],
    field_map: dict[str, str],
    *,
    row_number: int,
    density_g_cm3: float,
) -> dict[str, Any]:
    index = _read(row, field_map, "index") or str(row_number)
    salts = _split_list(_read_required(row, field_map, "salts"))
    concentrations = [_parse_percent(value) for value in _split_list(_read_required(row, field_map, "concentrations"))]
    concentration_type = _normalize_concentration_type(_read_required(row, field_map, "concentration_type"))
    total_atoms = int(float(_read_required(row, field_map, "total_atoms")))
    density = float(density_g_cm3)
    element_counts, formula_units, mole_fractions = composition_from_salts(
        salts, concentrations, concentration_type, total_atoms
    )

    return {
        "index": str(index).strip(),
        "salts": salts,
        "concentrations": concentrations,
        "concentration_type": concentration_type,
        "mole_fractions": mole_fractions,
        "total_atoms": total_atoms,
        "density_g_cm3": density,
        "formula_units": formula_units,
        "element_counts": element_counts,
    }


def _map_fields(fieldnames: list[str]) -> dict[str, str]:
    normalized = {_normalize_header(name): name for name in fieldnames}
    mapped: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            key = _normalize_header(alias)
            if key in normalized:
                mapped[canonical] = normalized[key]
                break
    missing = [
        name
        for name in ("salts", "concentrations", "concentration_type", "total_atoms")
        if name not in mapped
    ]
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
    return mapped


def _read(row: dict[str, str], field_map: dict[str, str], canonical: str) -> str:
    field = field_map.get(canonical)
    if not field:
        return ""
    return str(row.get(field, "")).strip()


def _read_required(row: dict[str, str], field_map: dict[str, str], canonical: str) -> str:
    value = _read(row, field_map, canonical)
    if not value:
        raise ValueError(f"CSV row is missing required value for {canonical}.")
    return value


def _split_list(text: str) -> list[str]:
    values = [part.strip() for part in _SPLIT_RE.split(text.strip()) if part.strip()]
    if not values:
        raise ValueError(f"Cannot parse list value: {text}")
    return values


def _parse_percent(text: str) -> float:
    value = text.strip().rstrip("%")
    return float(value)


def _normalize_concentration_type(text: str) -> str:
    compact = text.strip().lower().replace(" ", "")
    if compact in {"mol%", "mole%", "mol", "mole"}:
        return "mol%"
    if compact in {"wt%", "weight%", "wt", "weight"}:
        return "wt%"
    raise ValueError(f"Unsupported concentration type: {text}")


def _mole_fractions(
    formula_counts: list[dict[str, int]],
    concentrations: list[float],
    concentration_type: str,
) -> list[float]:
    if any(value < 0 for value in concentrations):
        raise ValueError("Concentrations must be non-negative.")
    if sum(concentrations) <= 0:
        raise ValueError("At least one concentration must be positive.")

    if concentration_type == "mol%":
        raw = concentrations
    elif concentration_type == "wt%":
        raw = [value / molar_mass(counts) for value, counts in zip(concentrations, formula_counts)]
    else:
        raise ValueError(f"Unsupported concentration type: {concentration_type}")

    total = sum(raw)
    return [value / total for value in raw]


def _round_units(mole_fractions: list[float], total_units: int) -> list[int]:
    raw = [fraction * total_units for fraction in mole_fractions]
    counts = [floor(value) for value in raw]
    positive = [idx for idx, fraction in enumerate(mole_fractions) if fraction > 0]
    if total_units >= len(positive):
        for idx in positive:
            if counts[idx] == 0:
                counts[idx] = 1

    delta = total_units - sum(counts)
    remainders = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - floor(raw[idx]), mole_fractions[idx]),
        reverse=True,
    )
    while delta > 0:
        for idx in remainders:
            counts[idx] += 1
            delta -= 1
            if delta == 0:
                break
    while delta < 0:
        for idx in reversed(remainders):
            if counts[idx] > (1 if idx in positive and total_units >= len(positive) else 0):
                counts[idx] -= 1
                delta += 1
                if delta == 0:
                    break
        else:
            break
    return counts


def _folder_name(system: dict[str, Any]) -> str:
    index = _safe_name(str(system["index"]))
    salts = _safe_name("-".join(system["salts"]))
    return f"{index}_{salts}"


def _safe_name(value: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip())
    return cleaned.strip("_") or "system"


def _normalize_header(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("（", "(")
        .replace("）", ")")
    )


def _write_metadata(
    output: Path,
    source_csv: Path,
    system: dict[str, Any],
    build_result: BoxBuildResult,
) -> None:
    metadata = {
        "source_csv": str(source_csv),
        "system": system,
        "box": {
            "data_file": str(build_result.output),
            "atom_count": build_result.atom_count,
            "box_length_angstrom": build_result.box_length_angstrom,
            "pdb_file": str(build_result.pdb_output) if build_result.pdb_output else None,
            "packmol_input": str(build_result.packmol_input) if build_result.packmol_input else None,
            "method": "ASE single-ion PDB + Packmol system packing + ASE LAMMPS-data conversion",
        },
    }
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
