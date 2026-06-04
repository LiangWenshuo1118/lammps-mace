from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess

from .chemistry import DEFAULT_MASSES


AMU_TO_GRAM = 1.66053906660e-24
CM3_TO_ANGSTROM3 = 1.0e24


@dataclass(frozen=True)
class BoxBuildResult:
    output: Path
    atom_count: int
    box_length_angstrom: float
    pdb_output: Path | None = None
    packmol_input: Path | None = None


def build_box_from_element_counts(
    element_counts: dict[str, int],
    density_g_cm3: float,
    output: str | Path,
    *,
    masses: dict[str, float] | None = None,
    packmol_tolerance_angstrom: float = 2.0,
    packmol_executable: str = "packmol",
    seed: int = 973,
) -> BoxBuildResult:
    """Build a mixed-composition LAMMPS data file from element counts.

    The high-throughput path follows the archived PDF workflow exactly:
    ASE generates single-ion PDB files, Packmol builds the multi-ion PDB box,
    and ASE converts that PDB to LAMMPS data.
    """
    species = [element for element, count in element_counts.items() if int(count) > 0]
    if not species:
        raise ValueError("At least one element count must be positive.")

    mass_table = DEFAULT_MASSES if masses is None else DEFAULT_MASSES | masses
    total_mass_amu = 0.0
    for element in species:
        if element not in mass_table:
            raise KeyError(f"Missing atomic mass for element '{element}'.")
        total_mass_amu += int(element_counts[element]) * float(mass_table[element])

    volume_a3 = total_mass_amu * AMU_TO_GRAM / float(density_g_cm3) * CM3_TO_ANGSTROM3
    box_length = volume_a3 ** (1.0 / 3.0)
    output_path = Path(output)
    artifact_dir = output_path.parent
    ion_dir = artifact_dir / "ions"
    pdb_output = artifact_dir / "box.pdb"
    packmol_input = artifact_dir / "packmol.inp"

    _write_single_ion_pdbs_with_ase(ion_dir, species)
    _write_packmol_input(
        packmol_input,
        ion_dir,
        pdb_output,
        element_counts,
        box_length,
        packmol_tolerance_angstrom,
        seed,
    )
    _run_packmol(packmol_input, artifact_dir, packmol_executable)
    atom_count = _write_lammps_data_from_pdb_with_ase(pdb_output, output_path, species, box_length)
    return BoxBuildResult(output_path, atom_count, box_length, pdb_output, packmol_input)


def _write_single_ion_pdbs_with_ase(output_dir: Path, species: list[str]) -> None:
    try:
        from ase import Atoms
    except ModuleNotFoundError as exc:
        raise RuntimeError("ASE is required for ion modeling. Install with: python -m pip install ase") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for element in species:
        Atoms(symbols=[element], positions=[(0.0, 0.0, 0.0)]).write(
            output_dir / f"{element}.pdb",
            format="proteindatabank",
        )


def _write_packmol_input(
    output: Path,
    ion_dir: Path,
    pdb_output: Path,
    element_counts: dict[str, int],
    box_length: float,
    tolerance: float,
    seed: int,
) -> None:
    lines = [
        f"tolerance {tolerance:.6g}",
        f"seed {int(seed)}",
        "",
        "filetype pdb",
        f"output {pdb_output.name}",
        f"pbc 0. 0. 0. {box_length:.6f} {box_length:.6f} {box_length:.6f}",
        "",
    ]
    for element, count in element_counts.items():
        if int(count) <= 0:
            continue
        lines.extend(
            [
                f"structure {ion_dir.name}/{element}.pdb",
                f"  number {int(count)}",
                f"  inside cube 0. 0. 0. {box_length:.6f}",
                "end structure",
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8")


def _run_packmol(packmol_input: Path, cwd: Path, packmol_executable: str) -> None:
    requested = Path(packmol_executable)
    if requested.parent != Path("."):
        executable = str(requested.expanduser().resolve())
        if not Path(executable).exists():
            executable = None
    else:
        executable = shutil.which(packmol_executable)
    if executable is None:
        raise RuntimeError(
            "Packmol is required for system modeling but was not found. "
            "Install it first, for example: conda install -c conda-forge packmol"
        )
    completed = subprocess.run(
        [executable, "-i", packmol_input.name],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    (cwd / "packmol.log").write_text(
        "STDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Packmol failed with exit code {completed.returncode}; see {cwd / 'packmol.log'}")


def _write_lammps_data_from_pdb_with_ase(
    pdb_input: Path,
    output: Path,
    species: list[str],
    box_length: float,
) -> int:
    try:
        from ase.io import read, write
    except ModuleNotFoundError as exc:
        raise RuntimeError("ASE is required for coordinate conversion. Install with: python -m pip install ase") from exc

    atoms = read(pdb_input, format="proteindatabank")
    atoms.set_cell([box_length, box_length, box_length])
    atoms.set_pbc(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write(
        output,
        atoms,
        format="lammps-data",
        specorder=species,
        masses=True,
        atom_style="atomic",
    )
    return len(atoms)
