import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from datasets import Dataset, DatasetDict
from PIL import Image


class NotebookMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(
            (Path(__file__).resolve().parents[1] / "01_defactify.ipynb").read_text()
        )
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["id"] == "split-metadata-diagnostic"
        )
        cls.namespace = {}
        definitions, switch, remainder = source.partition(
            "RUN_SPLIT_METADATA_DIAGNOSTICS = True"
        )
        cls.run_source = switch + remainder
        exec(definitions, cls.namespace)

    def fixture(self, png=False):
        splits = {}
        for split in ("train", "validation", "test"):
            images, sources, captions = [], [], []
            for source in (0, 1):
                for _ in range(2):
                    changed = split == "test" and source == 1
                    image = Image.new("RGB", (32, 8) if changed else (16, 16))
                    buffer = io.BytesIO()
                    image.save(
                        buffer,
                        format="PNG" if png and changed else "JPEG",
                        quality=30 if changed else 80,
                    )
                    images.append({"bytes": buffer.getvalue(), "path": None})
                    sources.append(source)
                    captions.append(f"same caption {source}")
            splits[split] = Dataset.from_dict(
                {"Image": images, "Label_B": sources, "Caption": captions}
            )
        return DatasetDict(splits)

    def compare(self, dataset, expected_total=12):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.namespace["compare_split_metadata"](
                dataset, {0: "real", 1: "generated"}, expected_total
            )

    def test_full_rows_group_counts_and_source_specific_drift(self):
        dataset = self.fixture()
        result = self.compare(dataset)
        self.assertEqual(len(result["metadata"]), 12)
        self.assertEqual(result["overview"]["images"].sum(), 12)
        self.assertTrue((result["overview"]["unique_captions"] == 1).all())
        self.assertTrue((result["overview"]["largest_source_group"] == 2).all())
        differences = result["categorical_drift"].set_index(
            ["source", "comparison", "feature"]
        )
        changed = differences.loc[("generated", "train vs test", "dimensions")]
        self.assertEqual(changed["total_variation"], 1.0)
        self.assertEqual(changed["right_unseen_in_left_fraction"], 1.0)
        self.assertEqual(
            differences.loc[("real", "train vs test", "dimensions"), "total_variation"],
            0.0,
        )
        numeric = result["numeric_drift"].set_index(["source", "comparison", "feature"])
        self.assertEqual(
            numeric.loc[
                ("generated", "train vs validation", "file_bytes"), "ks_distance"
            ],
            0.0,
        )
        # Casting images for inspection must not alter the original split schema.
        self.assertIsInstance(dataset["train"][0]["Image"], dict)

    def test_non_jpeg_metadata_and_full_dataset_guard(self):
        result = self.compare(self.fixture(png=True))
        png_rows = result["metadata"].query("format == 'PNG'")
        self.assertEqual(len(png_rows), 2)
        self.assertTrue((png_rows["jpeg_quantization_id"] == "unavailable").all())
        self.assertTrue((png_rows["jpeg_subsampling"] == "-1").all())
        with self.assertRaisesRegex(ValueError, "every expected dataset row"):
            self.compare(self.fixture(), expected_total=96_000)

    def test_cell_execution_saves_only_local_metadata_tables(self):
        namespace = dict(self.namespace)
        compare = namespace["compare_split_metadata"]
        namespace.update(
            raw=self.fixture(),
            audit={"invalid_rows": []},
            SOURCE_NAMES={0: "real", 1: "generated"},
            display=lambda *args: None,
            compare_split_metadata=lambda dataset, names: compare(dataset, names, 12),
        )
        with tempfile.TemporaryDirectory() as temporary:
            namespace["PROJECT_ROOT"] = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.run_source, namespace)
            output = Path(temporary) / "outputs" / "split_metadata"
            self.assertEqual(len(list(output.glob("*.csv"))), 6)
            self.assertTrue((output / "metadata.csv").exists())
            self.assertFalse((Path(temporary) / "checkpoints").exists())


if __name__ == "__main__":
    unittest.main()
