from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from lammps_mace.cli import build_parser, main


class CliTests(unittest.TestCase):
    def test_init_writes_example_csv_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            exit_code = main(["init", "--target", tmpdir])
            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(tmpdir) / "examples" / "systems.csv").exists())
            self.assertTrue((Path(tmpdir) / "outputs").is_dir())
            self.assertFalse((Path(tmpdir) / "outputs" / "boxes").exists())
            self.assertFalse((Path(tmpdir) / "examples" / "licl_973k.json").exists())

    def test_init_requires_target(self) -> None:
        with self.assertRaises(SystemExit):
            main(["init"])

    def test_build_boxes_uses_short_density_and_tolerance_options(self) -> None:
        args = build_parser().parse_args(
            [
                "build-boxes",
                "--csv",
                "examples/systems.csv",
                "--density",
                "1.5",
                "--packmol-tolerance",
                "2.5",
            ]
        )

        self.assertEqual(args.density_g_cm3, 1.5)
        self.assertEqual(args.packmol_tolerance_angstrom, 2.5)

    def test_build_boxes_requires_density(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["build-boxes", "--csv", "systems.csv"])

    def test_build_boxes_rejects_old_density_and_tolerance_options(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["build-boxes", "--csv", "systems.csv", "--density-g-cm3", "1.5"])
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["build-boxes", "--csv", "systems.csv", "--packmol-tolerance-angstrom", "2.5"]
            )

    def test_run_all_command_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            main(["run-all"])

    def test_mace_md_npt_command_renders_per_system_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            boxes_root = root / "boxes"
            system_folder = boxes_root / "001_LiCl"
            simulation_box = system_folder / "simulation_box"
            simulation_box.mkdir(parents=True)
            (simulation_box / "box.data").write_text("# minimal test placeholder\n", encoding="utf-8")
            (simulation_box / "system.json").write_text(
                json.dumps(
                    {
                        "system": {
                            "index": "001",
                            "element_counts": {"Li": 64, "Cl": 64},
                        }
                    }
                ),
                encoding="utf-8",
            )
            exit_code = main(
                [
                    "mace-md-npt",
                    "--boxes-root",
                    str(boxes_root),
                    "--model-path",
                    "models/base/model.pt",
                    "--temperature",
                    "973",
                    "1073",
                    "--pressure",
                    "0",
                    "--npt-steps",
                    "7",
                    "--thermo-every",
                    "5",
                    "--dump-every",
                    "5",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-npt" / "in.lammps").exists())
            self.assertTrue((system_folder / "1073K_00bar" / "mace-md-npt" / "in.lammps").exists())
            self.assertFalse((system_folder / "973K_00bar" / "mace-md-nvt").exists())
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-npt" / "run.sh").exists())
            self.assertTrue((boxes_root / "run_mace_md_npt_all.sh").exists())
            self.assertIn(
                "run 7",
                (system_folder / "973K_00bar" / "mace-md-npt" / "in.lammps").read_text(encoding="utf-8"),
            )
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-npt" / "md_settings.json").exists())
            self.assertFalse((system_folder / "mace-md-finetuned").exists())

    def test_mace_md_task_suffix_renders_separate_cli_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            boxes_root = root / "boxes"
            system_folder = boxes_root / "001_LiCl"
            simulation_box = system_folder / "simulation_box"
            simulation_box.mkdir(parents=True)
            (simulation_box / "box.data").write_text("# minimal test placeholder\n", encoding="utf-8")
            (simulation_box / "system.json").write_text(
                json.dumps(
                    {
                        "system": {
                            "index": "001",
                            "element_counts": {"Li": 64, "Cl": 64},
                        }
                    }
                ),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "mace-md-npt",
                    "--boxes-root",
                    str(boxes_root),
                    "--model-path",
                    "models/finetuned/model.pt",
                    "--temperature",
                    "973",
                    "--pressure",
                    "0",
                    "--task-suffix",
                    "finetuned",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-npt-finetuned" / "in.lammps").exists())
            self.assertFalse((system_folder / "973K_00bar" / "mace-md-npt").exists())
            self.assertTrue((boxes_root / "run_mace_md_npt_finetuned_all.sh").exists())

    def test_mace_md_nvt_command_renders_from_simulation_box_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            boxes_root = root / "boxes"
            system_folder = boxes_root / "001_LiCl"
            simulation_box = system_folder / "simulation_box"
            simulation_box.mkdir(parents=True)
            (simulation_box / "box.data").write_text("# minimal test placeholder\n", encoding="utf-8")
            (simulation_box / "system.json").write_text(
                json.dumps(
                    {
                        "system": {
                            "index": "001",
                            "element_counts": {"Li": 64, "Cl": 64},
                        }
                    }
                ),
                encoding="utf-8",
            )
            npt_dir = system_folder / "973K_00bar" / "mace-md-npt"
            npt_dir.mkdir(parents=True)
            (npt_dir / "in.lammps").write_text("# npt input\n", encoding="utf-8")

            exit_code = main(
                [
                    "mace-md-nvt",
                    "--boxes-root",
                    str(boxes_root),
                    "--model-path",
                    "models/base/model.pt",
                    "--nvt-steps",
                    "11",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-nvt" / "in.lammps").exists())
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-nvt" / "run.sh").exists())
            self.assertFalse((system_folder / "973K_00bar" / "mace-md-nvt" / "nvt_density.json").exists())
            self.assertTrue((boxes_root / "run_mace_md_nvt_all.sh").exists())
            self.assertIn(
                "read_data ../../simulation_box/box.data",
                (system_folder / "973K_00bar" / "mace-md-nvt" / "in.lammps").read_text(encoding="utf-8"),
            )

    def test_mace_md_nvt_command_can_use_npt_density_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            boxes_root = root / "boxes"
            system_folder = boxes_root / "001_LiCl"
            simulation_box = system_folder / "simulation_box"
            simulation_box.mkdir(parents=True)
            (simulation_box / "box.data").write_text("# minimal test placeholder\n", encoding="utf-8")
            (simulation_box / "system.json").write_text(
                json.dumps(
                    {
                        "system": {
                            "index": "001",
                            "element_counts": {"Li": 64, "Cl": 64},
                        }
                    }
                ),
                encoding="utf-8",
            )
            npt_dir = system_folder / "973K_00bar" / "mace-md-npt"
            npt_dir.mkdir(parents=True)
            (npt_dir / "in.lammps").write_text("# npt input\n", encoding="utf-8")
            (npt_dir / "final_mace_npt.data").write_text("# final npt data\n", encoding="utf-8")
            (npt_dir / "log.kk.half.lammps").write_text(
                "Step Temp PotEng KinEng TotEng Press Density Volume\n"
                "0 973 0 0 0 1 1.0 100\n"
                "100 973 0 0 0 1 2.0 90\n"
                "200 973 0 0 0 1 3.0 80\n"
                "300 973 0 0 0 1 4.0 70\n"
                "400 973 0 0 0 1 5.0 60\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "mace-md-nvt",
                    "--boxes-root",
                    str(boxes_root),
                    "--model-path",
                    "models/base/model.pt",
                    "--nvt-steps",
                    "11",
                    "--use-npt-density",
                    "0.8",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((system_folder / "973K_00bar" / "mace-md-nvt" / "nvt_density.json").exists())
            self.assertIn(
                "change_box all x final 0.0",
                (system_folder / "973K_00bar" / "mace-md-nvt" / "in.lammps").read_text(encoding="utf-8"),
            )

    def test_removed_commands_are_rejected(self) -> None:
        for command in ("mace-md", "mace-md-finetuned", "train-command"):
            with self.assertRaises(SystemExit):
                main([command])

    def test_steps_argument_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "mace-md-npt",
                    "--boxes-root",
                    "boxes",
                    "--model-path",
                    "models/base/model.pt",
                    "--steps",
                    "10",
                ]
            )

    def test_finetune_command_prepares_generic_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data" / "finetune"
            dataset.mkdir(parents=True)
            for filename in ("train.extxyz", "valid.extxyz"):
                (dataset / filename).write_text("", encoding="utf-8")

            task_dir = root / "outputs" / "finetune-mace"
            exit_code = main(
                [
                    "finetune",
                    "--task-dir",
                    str(task_dir),
                    "--dataset-root",
                    str(dataset),
                    "--foundation-model",
                    str(root / "models" / "base" / "mace-mp-0b3-medium.model"),
                    "--max-num-epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--require-dataset",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((task_dir / "finetune_settings.json").exists())
            self.assertTrue((task_dir / "run_train.sh").exists())
            settings = json.loads((task_dir / "finetune_settings.json").read_text(encoding="utf-8"))
            self.assertNotIn("name", settings["dataset"])
            self.assertEqual(settings["dataset"]["root"], str(dataset.resolve()))
            self.assertIn("train.extxyz", settings["command"])
            self.assertIn("--name mace-mp-0b3-medium-finetuned", settings["command"])
            self.assertIn("--forces_key forces", settings["command"])
            self.assertNotIn("train", settings)
            self.assertNotIn("command_text", settings)

    def test_finetune_command_exposes_train_command_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data" / "finetune"
            dataset.mkdir(parents=True)
            for filename in ("train.extxyz", "valid.extxyz", "test.extxyz"):
                (dataset / filename).write_text("", encoding="utf-8")

            task_dir = root / "outputs" / "finetune-mace"
            exit_code = main(
                [
                    "finetune",
                    "--task-dir",
                    str(task_dir),
                    "--dataset-root",
                    str(dataset),
                    "--test-file",
                    "test.extxyz",
                    "--results-dir",
                    str(root / "custom-results"),
                    "--work-dir",
                    str(root / "custom-work"),
                    "--log-dir",
                    str(root / "custom-logs"),
                    "--model-dir",
                    str(root / "custom-models"),
                    "--checkpoints-dir",
                    str(root / "custom-checkpoints"),
                    "--r-max",
                    "6.0",
                    "--valid-batch-size",
                    "3",
                    "--require-dataset",
                ]
            )

            self.assertEqual(exit_code, 0)
            command = json.loads((task_dir / "finetune_settings.json").read_text(encoding="utf-8"))["command"]
            self.assertIn("--test_file", command)
            self.assertIn(str((dataset / "test.extxyz").resolve()), command)
            self.assertIn("--results_dir", command)
            self.assertIn(str((root / "custom-results").resolve()), command)
            self.assertIn("--work_dir", command)
            self.assertIn(str((root / "custom-work").resolve()), command)
            self.assertIn("--log_dir", command)
            self.assertIn(str((root / "custom-logs").resolve()), command)
            self.assertIn("--model_dir", command)
            self.assertIn(str((root / "custom-models").resolve()), command)
            self.assertIn("--checkpoints_dir", command)
            self.assertIn(str((root / "custom-checkpoints").resolve()), command)
            self.assertIn("--r_max 6.0", command)
            self.assertIn("--valid_batch_size 3", command)

    def test_finetune_dataset_name_option_is_removed(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["finetune", "--dataset-name", "custom"])


if __name__ == "__main__":
    unittest.main()
