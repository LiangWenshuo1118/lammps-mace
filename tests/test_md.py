from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from lammps_mace.md import (
    NPT_RUN_LABEL,
    aggregate_run_script_name,
    render_npt_inputs_for_boxes,
    render_nvt_inputs_for_boxes,
)


class MdInputTests(unittest.TestCase):
    def test_mace_md_npt_generates_temperature_pressure_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            system_folder = self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})

            results = render_npt_inputs_for_boxes(
                {"npt_steps": 7, "thermo_every": 5, "dump_every": 5},
                root,
                temperatures_k=[973, 1073],
                pressures_bar=[0, 1],
                model_path="models/base/model.pt",
            )

            self.assertEqual(len(results), 4)
            npt_input = system_folder / "973K_00bar" / "mace-md-npt" / "in.lammps"
            run_script = system_folder / "973K_00bar" / "mace-md-npt" / "run.sh"
            aggregate_script = root / "run_mace_md_npt_all.sh"
            self.assertTrue(npt_input.exists())
            self.assertTrue((system_folder / "1073K_01bar" / "mace-md-npt" / "in.lammps").exists())
            self.assertTrue(run_script.exists())
            self.assertTrue(aggregate_script.exists())

            npt_text = npt_input.read_text(encoding="utf-8")
            aggregate_script_text = aggregate_script.read_text(encoding="utf-8")
            self.assertIn("read_data ../../simulation_box/box.data", npt_text)
            self.assertIn("pair_coeff * * Li Cl", npt_text)
            self.assertIn("fix integrator all npt temp 973.0 973.0", npt_text)
            self.assertIn("iso 0.0000 0.0000", npt_text)
            self.assertIn("run 7", npt_text)
            self.assertIn("write_data final_mace_npt.data", npt_text)
            self.assertIn('bash "001_LiCl/973K_00bar/mace-md-npt/run.sh"', aggregate_script_text)
            self.assertIn('bash "001_LiCl/1073K_01bar/mace-md-npt/run.sh"', aggregate_script_text)

    def test_mace_md_nvt_follows_existing_npt_task_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            system_folder = self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})
            render_npt_inputs_for_boxes(
                {"npt_steps": 7, "thermo_every": 5, "dump_every": 5},
                root,
                temperatures_k=[973, 1073],
                pressures_bar=[0],
                model_path="models/base/model.pt",
            )

            results = render_nvt_inputs_for_boxes(
                {"nvt_steps": 11, "thermo_every": 5, "dump_every": 5},
                root,
                model_path="models/base/model.pt",
            )

            self.assertEqual(len(results), 2)
            nvt_input = system_folder / "973K_00bar" / "mace-md-nvt" / "in.lammps"
            aggregate_script = root / "run_mace_md_nvt_all.sh"
            self.assertTrue(nvt_input.exists())
            self.assertTrue((system_folder / "1073K_00bar" / "mace-md-nvt" / "in.lammps").exists())
            self.assertTrue(aggregate_script.exists())
            nvt_text = nvt_input.read_text(encoding="utf-8")
            self.assertIn("read_data ../../simulation_box/box.data", nvt_text)
            self.assertNotIn("change_box all", nvt_text)
            self.assertIn("fix integrator all nvt temp 973.0 973.0", nvt_text)
            self.assertIn("run 11", nvt_text)
            self.assertIn(
                'bash "001_LiCl/973K_00bar/mace-md-nvt/run.sh"',
                aggregate_script.read_text(encoding="utf-8"),
            )

    def test_mace_md_nvt_can_filter_temperatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            system_folder = self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})
            render_npt_inputs_for_boxes({}, root, temperatures_k=[973, 1073], pressures_bar=[0])

            results = render_nvt_inputs_for_boxes({}, root, temperatures_k=[1073])

            self.assertEqual(len(results), 1)
            self.assertFalse((system_folder / "973K_00bar" / "mace-md-nvt").exists())
            self.assertTrue((system_folder / "1073K_00bar" / "mace-md-nvt" / "in.lammps").exists())

    def test_mace_md_nvt_uses_average_density_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            system_folder = self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})
            render_npt_inputs_for_boxes({}, root, temperatures_k=[973], pressures_bar=[0])
            npt_dir = system_folder / "973K_00bar" / "mace-md-npt"
            (npt_dir / f"final_{NPT_RUN_LABEL}.data").write_text("# final npt data\n", encoding="utf-8")
            (npt_dir / "log.kk.half.lammps").write_text(
                "\n".join(
                    [
                        "Step Temp PotEng KinEng TotEng Press Density Volume",
                        "0 973 0 0 0 1 1.0 100",
                        "100 973 0 0 0 1 2.0 90",
                        "200 973 0 0 0 1 3.0 80",
                        "300 973 0 0 0 1 4.0 70",
                        "400 973 0 0 0 1 5.0 60",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            results = render_nvt_inputs_for_boxes(
                {"nvt_steps": 11, "thermo_every": 5, "dump_every": 5},
                root,
                model_path="models/custom.pt",
                use_npt_density_fraction=0.8,
            )

            self.assertEqual(len(results), 1)
            nvt_input = system_folder / "973K_00bar" / "mace-md-nvt" / "in.lammps"
            density_metadata = system_folder / "973K_00bar" / "mace-md-nvt" / "nvt_density.json"
            self.assertTrue(nvt_input.exists())
            self.assertTrue(density_metadata.exists())

            nvt_text = nvt_input.read_text(encoding="utf-8")
            metadata = json.loads(density_metadata.read_text(encoding="utf-8"))
            self.assertEqual(metadata["sample_count"], 5)
            self.assertEqual(metadata["used_sample_count"], 4)
            self.assertAlmostEqual(metadata["average_density_g_cm3"], 3.5)
            self.assertIn("read_data ../mace-md-npt/final_mace_npt.data", nvt_text)
            self.assertIn("average NPT density 3.50000000 g/cm^3", nvt_text)
            self.assertIn("change_box all x final 0.0", nvt_text)

    def test_mace_md_task_suffix_keeps_finetuned_tasks_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            system_folder = self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})

            render_npt_inputs_for_boxes(
                {},
                root,
                temperatures_k=[973],
                pressures_bar=[0],
                model_path="models/finetuned/model.pt",
                task_suffix="finetuned",
            )
            npt_dir = system_folder / "973K_00bar" / "mace-md-npt-finetuned"
            (npt_dir / f"final_{NPT_RUN_LABEL}.data").write_text("# final npt data\n", encoding="utf-8")
            (npt_dir / "log.kk.half.lammps").write_text(
                "\n".join(
                    [
                        "Step Temp PotEng KinEng TotEng Press Density Volume",
                        "0 973 0 0 0 1 2.0 100",
                        "100 973 0 0 0 1 4.0 90",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            render_nvt_inputs_for_boxes(
                {},
                root,
                model_path="models/finetuned/model.pt",
                use_npt_density_fraction=1.0,
                task_suffix="finetuned",
            )

            nvt_input = system_folder / "973K_00bar" / "mace-md-nvt-finetuned" / "in.lammps"
            self.assertTrue(npt_dir.exists())
            self.assertTrue(nvt_input.exists())
            self.assertFalse((system_folder / "973K_00bar" / "mace-md-npt").exists())
            self.assertFalse((system_folder / "973K_00bar" / "mace-md-nvt").exists())
            self.assertTrue((root / "run_mace_md_npt_finetuned_all.sh").exists())
            self.assertTrue((root / "run_mace_md_nvt_finetuned_all.sh").exists())
            self.assertIn(
                "read_data ../mace-md-npt-finetuned/final_mace_npt.data",
                nvt_input.read_text(encoding="utf-8"),
            )

    def test_mace_md_nvt_requires_existing_npt_task_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})

            with self.assertRaisesRegex(ValueError, "No matching NPT task folders"):
                render_nvt_inputs_for_boxes({}, root)

    def test_mace_md_nvt_requires_completed_npt_when_npt_density_is_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "boxes"
            self._write_system(root, "001_LiCl", {"Li": 64, "Cl": 64})
            render_npt_inputs_for_boxes({}, root, temperatures_k=[973], pressures_bar=[0])

            with self.assertRaisesRegex(ValueError, "Missing NPT final data"):
                render_nvt_inputs_for_boxes({}, root, use_npt_density_fraction=0.8)

    def _write_system(self, root: Path, folder_name: str, element_counts: dict[str, int]) -> Path:
        system_folder = root / folder_name
        simulation_box = system_folder / "simulation_box"
        simulation_box.mkdir(parents=True)
        (simulation_box / "box.data").write_text("# minimal test placeholder\n", encoding="utf-8")
        (simulation_box / "system.json").write_text(
            json.dumps(
                {
                    "system": {
                        "index": folder_name.split("_", 1)[0],
                        "element_counts": element_counts,
                    }
                }
            ),
            encoding="utf-8",
        )
        return system_folder


if __name__ == "__main__":
    unittest.main()
