import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from defactify_training import binary_metrics, export_final_model, train_experiment


class TinyDataset(Dataset):
    def __init__(self):
        self.rows = {"Label_A": [0, 1, 0, 1]}

    def __len__(self):
        return 4

    def __getitem__(self, index):
        label = self.rows["Label_A"][index]
        return (
            torch.tensor([float(index % 2)]),
            torch.tensor(float(label)),
            torch.tensor(label, dtype=torch.long),
        )


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, images):
        return self.linear(images).squeeze(-1), None


class TrainingTests(unittest.TestCase):
    def test_binary_metrics_report_balanced_error_and_confusion(self):
        result = binary_metrics([0, 0, 1, 1], [-2, 1, -1, 2])
        self.assertAlmostEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["balanced_error_rate"], 0.5)
        self.assertEqual(result["confusion_matrix_real_ai"], [[1, 1], [1, 1]])

    def test_epoch_csv_checkpoints_resume_and_export(self):
        torch.manual_seed(1)
        loader = DataLoader(TinyDataset(), batch_size=2)
        metadata = {
            "dataset_revision": "synthetic",
            "seed": 1,
            "split_sizes": {"train": 4, "validation": 4, "test": 4},
            "image_size": 224,
            "split_report": {"split_fingerprint_sha256": "synthetic-fingerprint"},
            "post_repair_audit": {
                "caption_overlap_groups": 0,
                "byte_identical_overlap_groups": 0,
                "invalid_row_count": 0,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kwargs = dict(
                model_name="rgb_baseline",
                model_config={},
                train_loader=loader,
                validation_loader=loader,
                device=torch.device("cpu"),
                epochs=2,
                learning_rate=0.01,
                auxiliary_weight=0.0,
                metadata=metadata,
                checkpoints_dir=root / "checkpoints",
                outputs_dir=root / "outputs",
            )
            model, history, score = train_experiment(TinyDetector(), **kwargs)
            self.assertEqual(len(history), 2)
            csv = pd.read_csv(root / "outputs/epochs_rgb_baseline.csv")
            self.assertEqual(csv["epoch"].tolist(), [1, 2])
            self.assertIn("validation_balanced_accuracy", csv)
            self.assertTrue((root / "checkpoints/rgb_baseline/best.pt").exists())
            self.assertTrue((root / "checkpoints/rgb_baseline/last.pt").exists())
            (root / "outputs/epochs_rgb_baseline.csv").unlink()
            resumed, resumed_history, resumed_score = train_experiment(
                TinyDetector(), **kwargs
            )
            self.assertEqual(len(resumed_history), 2)
            self.assertEqual(score, resumed_score)
            self.assertIsInstance(resumed, TinyDetector)
            self.assertTrue((root / "outputs/epochs_rgb_baseline.csv").exists())
            name, bundle_path, info = export_final_model(
                root / "checkpoints",
                ["rgb_baseline"],
                2,
                root / "outputs",
                {0: "real", 1: "AI"},
            )
            self.assertEqual(name, "rgb_baseline")
            self.assertEqual(info["validation_balanced_accuracy"], score)
            bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
            self.assertEqual(bundle["model_name"], name)
            self.assertTrue((root / "outputs/final_weights.pt").exists())
            self.assertTrue((root / "outputs/final_model.json").exists())
            best_path = root / "checkpoints/rgb_baseline/best.pt"
            stale = torch.load(best_path, map_location="cpu", weights_only=True)
            stale["run_id"] = "another-run"
            torch.save(stale, best_path)
            with self.assertRaisesRegex(
                ValueError, "Stale or mismatched best checkpoint"
            ):
                export_final_model(
                    root / "checkpoints",
                    ["rgb_baseline"],
                    2,
                    root / "outputs",
                    {0: "real", 1: "AI"},
                )


if __name__ == "__main__":
    unittest.main()
