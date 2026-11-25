import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from defactify_models import (
    DETECTOR_CAUTION,
    MODEL_RGB_BASELINE,
    MODEL_RGB_FOURIER_ATTENTION_AUX,
    MODEL_RGB_FOURIER_CONCAT,
    RGBBaseline,
    TwoBranchDetector,
    build_detector,
    load_detector,
    make_detector_checkpoint,
    predict_ai_probability,
    prediction_report,
    prepare_image,
)


class DefactifyModelTests(unittest.TestCase):
    def test_architectures_match_expected_output_shapes(self):
        sample = torch.rand(2, 3, 64, 64)
        cases = [
            (RGBBaseline(pretrained=False), False),
            (TwoBranchDetector("concat", pretrained=False), False),
            (TwoBranchDetector("attention", auxiliary=True, pretrained=False), True),
        ]
        with torch.no_grad():
            for model, has_auxiliary_head in cases:
                binary, source = model.eval()(sample)
                self.assertEqual(binary.shape, (2,))
                if has_auxiliary_head:
                    self.assertEqual(source.shape, (2, 6))
                else:
                    self.assertIsNone(source)

    def test_factory_supports_only_declared_experiment_names(self):
        self.assertIsInstance(build_detector(MODEL_RGB_BASELINE), RGBBaseline)
        self.assertEqual(build_detector(MODEL_RGB_FOURIER_CONCAT).fusion, "concat")
        attention_model = build_detector(MODEL_RGB_FOURIER_ATTENTION_AUX)
        self.assertEqual(attention_model.fusion, "attention")
        self.assertIsNotNone(attention_model.source_head)
        with self.assertRaises(ValueError):
            build_detector("unknown_model")
        with self.assertRaises(ValueError):
            build_detector(MODEL_RGB_FOURIER_CONCAT, {"fusion": "attention"})

    def test_prepare_image_returns_batched_rgb_tensor(self):
        image = Image.new("L", (17, 13), color=128)
        tensor = prepare_image(image, image_size=32)
        self.assertEqual(tensor.shape, (1, 3, 32, 32))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_checkpoint_round_trip_loads_eval_model_and_predicts(self):
        model = build_detector(MODEL_RGB_BASELINE)
        checkpoint = make_detector_checkpoint(
            model,
            MODEL_RGB_BASELINE,
            image_size=64,
            threshold=0.5,
            source_names={0: "real", 1: "AI-generated"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "detector.pt"
            torch.save(checkpoint, path)
            loaded = load_detector(path)

        self.assertEqual(loaded.model_name, MODEL_RGB_BASELINE)
        self.assertFalse(loaded.model.training)
        image = Image.new("RGB", (64, 64), color=(80, 120, 160))
        probability = loaded.predict_proba(image)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        report = prediction_report(loaded, image)
        self.assertIn("ai_probability", report)
        self.assertIn("not proof", report["caution"])
        self.assertEqual(report["caution"], DETECTOR_CAUTION)

    def test_probability_helper_restores_training_mode(self):
        model = build_detector(MODEL_RGB_BASELINE)
        model.train()
        image = Image.new("RGB", (64, 64), color="white")
        probability = predict_ai_probability(model, image, image_size=64)
        self.assertTrue(model.training)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_loader_rejects_bad_checkpoint_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.pt"
            torch.save({"schema_version": 999, "model_name": MODEL_RGB_BASELINE}, path)
            with self.assertRaises(ValueError):
                load_detector(path)


if __name__ == "__main__":
    unittest.main()
