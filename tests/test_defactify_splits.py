import unittest

from defactify_splits import group_assignments


class GroupAssignmentsTests(unittest.TestCase):
    def test_transitive_caption_and_hash_links_stay_together(self):
        sizes = {"train": 2, "validation": 2, "test": 2}
        records = [
            ("train", 0, "shared caption", "hash 0", 0),
            ("train", 1, "other caption", "hash 1", 1),
            ("validation", 0, "shared caption", "hash 2", 0),
            ("validation", 1, "independent", "hash 3", 1),
            ("test", 0, "third caption", "hash 2", 0),
            ("test", 1, "last caption", "hash 4", 1),
        ]
        indices, report = group_assignments(records, sizes, seed=42)
        destination = {
            index: split for split, members in indices.items() for index in members
        }
        self.assertEqual(len(destination), len(records))
        self.assertEqual(set(destination), set(range(len(records))))
        self.assertEqual(destination[0], destination[2])  # shared caption
        self.assertEqual(destination[2], destination[4])  # shared hash, transitively
        self.assertEqual(sum(report["split_sizes"].values()), len(records))
        self.assertEqual((indices, report), group_assignments(records, sizes, seed=42))

    def test_clean_split_keeps_every_original_row(self):
        sizes = {"train": 2, "validation": 1, "test": 1}
        records = [
            ("train", 0, "a", "1", 0),
            ("train", 1, "b", "2", 1),
            ("validation", 0, "c", "3", 0),
            ("test", 0, "d", "4", 1),
        ]
        indices, report = group_assignments(records, sizes)
        self.assertEqual(indices, {"train": [0, 1], "validation": [2], "test": [3]})
        self.assertEqual(report["moved_rows"], 0)


if __name__ == "__main__":
    unittest.main()
