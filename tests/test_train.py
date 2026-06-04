from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from lammps_mace.train import prepare_finetune_task, shell_command


class TrainTaskTests(unittest.TestCase):
    def test_prepare_finetune_task_writes_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data" / "finetune"
            dataset.mkdir(parents=True)
            for filename in ("train.extxyz", "valid.extxyz"):
                (dataset / filename).write_text("", encoding="utf-8")
            foundation = root / "models" / "base" / "mace-mp-0b3-medium.model"
            foundation.parent.mkdir(parents=True)
            foundation.write_text("placeholder\n", encoding="utf-8")

            result = prepare_finetune_task(
                task_dir=root / "outputs" / "finetune-mace",
                dataset_root=dataset,
                foundation_model=foundation,
                max_num_epochs=1,
                batch_size=2,
                require_dataset=True,
            )

            self.assertTrue(result.settings_file.exists())
            self.assertTrue(result.run_script.exists())
            self.assertTrue((result.task_dir / "models").is_dir())
            self.assertTrue((result.task_dir / "logs").is_dir())
            self.assertEqual(result.missing_dataset_files, ())

            settings = json.loads(result.settings_file.read_text(encoding="utf-8"))
            self.assertNotIn("name", settings["dataset"])
            self.assertEqual(settings["dataset"]["source_train_file"], str((dataset / "train.extxyz").resolve()))
            self.assertEqual(settings["dataset"]["source_valid_file"], str((dataset / "valid.extxyz").resolve()))
            self.assertEqual(settings["run_script"], "run_train.sh")
            self.assertNotIn("train", settings)
            self.assertNotIn("command_text", settings)
            self.assertIn(str((dataset / "train.extxyz").resolve()), settings["command"])
            self.assertIn(str((dataset / "valid.extxyz").resolve()), settings["command"])
            self.assertIn("--foundation_model", result.command)
            self.assertIn(str(foundation.resolve()), result.command)
            command_text = shell_command(result.command)
            self.assertIn("--name mace-mp-0b3-medium-finetuned", command_text)
            self.assertIn("--multiheads_finetuning False", command_text)
            self.assertIn("--E0s average", command_text)
            self.assertIn("--energy_weight 1.0", command_text)
            self.assertIn("--forces_weight 1.0", command_text)
            self.assertIn("--lr 0.01", command_text)
            self.assertIn("--scaling rms_forces_scaling", command_text)
            self.assertIn("--ema --ema_decay 0.99 --amsgrad", command_text)
            self.assertIn("--default_dtype float64", command_text)
            self.assertIn("--seed 3", command_text)
            self.assertIn("mace_run_train", shell_command(result.command))

    def test_output_dir_controls_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data" / "finetune"
            dataset.mkdir(parents=True)
            for filename in ("train.extxyz", "valid.extxyz"):
                (dataset / filename).write_text("", encoding="utf-8")
            model_dir = root / "models" / "finetuned"

            result = prepare_finetune_task(
                task_dir=root / "outputs" / "finetune",
                dataset_root=dataset,
                require_dataset=True,
                extra_train_settings={"output_dir": str(model_dir)},
            )

            settings = json.loads(result.settings_file.read_text(encoding="utf-8"))
            self.assertIn(str(model_dir.resolve()), settings["command"])
            self.assertIn("--model_dir", result.command)
            self.assertIn(str(model_dir.resolve()), result.command)
            self.assertIn("--results_dir", result.command)
            self.assertIn(str((result.task_dir / "results").resolve()), result.command)

    def test_prepare_finetune_task_can_write_smoke_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "data" / "finetune"
            dataset.mkdir(parents=True)
            train_source = dataset / "train.extxyz"
            valid_source = dataset / "valid.extxyz"
            train_source.write_text(_extxyz_frames(3), encoding="utf-8")
            valid_source.write_text(_extxyz_frames(2), encoding="utf-8")

            result = prepare_finetune_task(
                task_dir=root / "outputs" / "finetune-smoke",
                dataset_root=dataset,
                smoke_max_configs=1,
                require_dataset=True,
            )

            self.assertIsNotNone(result.smoke_dataset_dir)
            smoke_train = result.task_dir / "smoke-dataset" / "train.extxyz"
            smoke_valid = result.task_dir / "smoke-dataset" / "valid.extxyz"
            self.assertEqual(smoke_train.read_text(encoding="utf-8").count("energy="), 1)
            self.assertEqual(smoke_valid.read_text(encoding="utf-8").count("energy="), 1)

            settings = json.loads(result.settings_file.read_text(encoding="utf-8"))
            self.assertEqual(settings["dataset"]["source_train_file"], str(train_source.resolve()))
            self.assertEqual(settings["dataset"]["smoke_max_configs"], 1)
            self.assertEqual(settings["dataset"]["smoke_dataset_dir"], str((result.task_dir / "smoke-dataset").resolve()))
            self.assertIn(str(smoke_train.resolve()), settings["command"])

def _extxyz_frames(count: int) -> str:
    frame = (
        "2\n"
        'Properties=species:S:1:pos:R:3 energy=0.0 forces="0 0 0 0 0 0"\n'
        "Li 0.0 0.0 0.0\n"
        "Cl 1.0 0.0 0.0\n"
    )
    return frame * count


if __name__ == "__main__":
    unittest.main()
