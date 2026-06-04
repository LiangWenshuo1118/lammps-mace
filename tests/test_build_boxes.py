from __future__ import annotations

import tempfile
from pathlib import Path
import sys
import unittest

from lammps_mace.build_boxes import build_boxes_from_csv, composition_from_salts


class BuildBoxesTests(unittest.TestCase):
    def test_composition_from_mol_percent_salts(self) -> None:
        element_counts, formula_units, mole_fractions = composition_from_salts(
            ["LiCl", "KCl"], [60, 40], "mol%", 128
        )
        self.assertEqual(sum(element_counts.values()), 128)
        self.assertEqual(element_counts["Cl"], element_counts["Li"] + element_counts["K"])
        self.assertAlmostEqual(sum(mole_fractions), 1.0)
        self.assertEqual(set(formula_units), {"LiCl", "KCl"})

    def test_build_boxes_creates_one_folder_per_csv_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_packmol = Path(tmpdir) / "fake_packmol.py"
            fake_packmol.write_text(
                f"""#!{sys.executable}
import pathlib
import re
import sys
from ase import Atoms

if "-i" in sys.argv:
    text = pathlib.Path(sys.argv[sys.argv.index("-i") + 1]).read_text(encoding="utf-8")
else:
    text = sys.stdin.read()
output = re.search(r"^output\\s+(\\S+)", text, re.MULTILINE).group(1)
atoms = []
current = None
for line in text.splitlines():
    structure = re.match(r"structure\\s+\\S+/([A-Za-z][a-z]?)\\.pdb", line.strip())
    if structure:
        current = structure.group(1)
        continue
    number = re.match(r"number\\s+(\\d+)", line.strip())
    if number and current:
        atoms.extend([current] * int(number.group(1)))

positions = [(float(idx % 10), float((idx // 10) % 10), float((idx // 100) % 10)) for idx in range(len(atoms))]
Atoms(symbols=atoms, positions=positions).write(pathlib.Path(output), format="proteindatabank")
""",
                encoding="utf-8",
            )
            fake_packmol.chmod(0o755)
            csv_path = Path(tmpdir) / "systems.csv"
            csv_path.write_text(
                "system_index,salts,concentrations,concentration_type,total_atoms\n"
                "001,LiCl-KCl,60-40,mol%,128\n"
                "002,NaCl-KCl,55-45,wt%,128\n",
                encoding="utf-8",
            )
            output_root = Path(tmpdir) / "boxes"

            results = build_boxes_from_csv(
                csv_path,
                output_root,
                density_g_cm3=1.46,
                packmol_executable=str(fake_packmol),
            )

            self.assertEqual(len(results), 2)
            for result in results:
                simulation_box = result.folder / "simulation_box"
                self.assertTrue(result.data_file.exists())
                self.assertTrue(result.metadata_file.exists())
                self.assertEqual(result.data_file, simulation_box / "box.data")
                self.assertEqual(result.metadata_file, simulation_box / "system.json")
                self.assertTrue((simulation_box / "box.pdb").exists())
                self.assertTrue((simulation_box / "packmol.inp").exists())
                self.assertTrue((simulation_box / "packmol.log").exists())
                self.assertTrue(
                    (simulation_box / "ions" / "Li.pdb").exists()
                    or (simulation_box / "ions" / "Na.pdb").exists()
                )
                self.assertFalse((result.folder / "box.data").exists())
                self.assertFalse((result.folder / "system.json").exists())
                self.assertFalse((result.folder / "mace-md").exists())
                self.assertFalse((result.folder / "mace-md-finetuned").exists())
                self.assertEqual(result.atom_count, 128)
            self.assertTrue((output_root / "001_LiCl-KCl").is_dir())
            self.assertTrue((output_root / "002_NaCl-KCl").is_dir())

    def test_total_atoms_must_be_in_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "systems.csv"
            csv_path.write_text(
                "system_index,salts,concentrations,concentration_type\n"
                "001,LiCl,100,mol%\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "total_atoms"):
                build_boxes_from_csv(csv_path, Path(tmpdir) / "boxes", density_g_cm3=1.46)


if __name__ == "__main__":
    unittest.main()
