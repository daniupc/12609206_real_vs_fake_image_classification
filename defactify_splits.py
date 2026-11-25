"""Audit Defactify and repair leakage in its published splits without dropping rows."""

from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import random

from datasets import DatasetDict, Image as HFImage, concatenate_datasets
from PIL import Image
import pandas as pd


def normalize_caption(value):
    return " ".join(str(value).casefold().split())


def image_bytes(field):
    return (
        field["bytes"]
        if field["bytes"] is not None
        else Path(field["path"]).read_bytes()
    )


def audit_dataset(dataset, split_names, source_names):
    """Return an audit summary and row metadata in split_names/row order."""
    counts, formats, dimensions = Counter(), Counter(), Counter()
    caption_splits, hash_splits = defaultdict(set), defaultdict(set)
    caption_rows, hash_rows = Counter(), Counter()
    invalid, records = [], []
    for split in split_names:
        rows = dataset[split].cast_column("Image", HFImage(decode=False))
        for i, row in enumerate(rows):
            binary, source = int(row["Label_A"]), int(row["Label_B"])
            if (
                binary not in (0, 1)
                or source not in source_names
                or binary != int(source != 0)
            ):
                invalid.append((split, i, binary, source))
            counts[(split, binary, source)] += 1
            caption = normalize_caption(row["Caption"])
            blob = image_bytes(row["Image"])
            digest = sha256(blob).hexdigest()
            records.append((split, i, caption, digest, source))
            caption_splits[caption].add(split)
            hash_splits[digest].add(split)
            caption_rows[caption] += 1
            hash_rows[digest] += 1
            try:
                with Image.open(BytesIO(blob)) as image:
                    formats[(source, image.format)] += 1
                    dimensions[(source, image.size)] += 1
                    image.verify()
            except Exception as exc:
                invalid.append((split, i, f"bad image: {exc}"))
    cross_captions = {key for key, splits in caption_splits.items() if len(splits) > 1}
    cross_hashes = {key for key, splits in hash_splits.items() if len(splits) > 1}
    summary = {
        "counts": pd.DataFrame(
            [(s, b, src, n) for (s, b, src), n in sorted(counts.items())],
            columns=["split", "binary", "source", "count"],
        ),
        "formats_by_source": pd.DataFrame(
            [
                (source_names.get(src, str(src)), fmt, n)
                for (src, fmt), n in sorted(formats.items())
            ],
            columns=["source", "format", "count"],
        ),
        "caption_overlap_groups": len(cross_captions),
        "caption_overlap_rows": sum(caption_rows[key] for key in cross_captions),
        "byte_identical_overlap_groups": len(cross_hashes),
        "byte_identical_overlap_rows": sum(hash_rows[key] for key in cross_hashes),
        "invalid_rows": invalid,
        "unique_dimensions_by_source": {
            source_names[src]: len({size for (s, size) in dimensions if s == src})
            for src in source_names
        },
    }
    return summary, records


def group_assignments(records, expected_sizes, seed=42):
    """Assign connected caption/hash groups while minimizing changed origins."""
    splits = tuple(expected_sizes)
    if len(records) != sum(expected_sizes.values()):
        raise ValueError("The input row count does not match the requested split sizes")
    parent = list(range(len(records)))
    rank = [0] * len(records)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    first_caption, first_hash = {}, {}
    for i, (split, row, caption, digest, source) in enumerate(records):
        for owners, key in ((first_caption, caption), (first_hash, digest)):
            if key in owners:
                union(i, owners[key])
            else:
                owners[key] = i

    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    origins = {
        root: Counter(records[i][0] for i in members)
        for root, members in groups.items()
    }
    rng = random.Random(seed)
    roots = list(groups)
    rng.shuffle(roots)
    assignment = {}
    sizes = Counter()
    for root in roots:
        # Keep each group where most of its original rows were. A seeded order
        # and the current size deficit make tied groups reproducible.
        split = max(
            splits,
            key=lambda name: (origins[root][name], expected_sizes[name] - sizes[name]),
        )
        assignment[root] = split
        sizes[split] += len(groups[root])

    indices = {split: [] for split in splits}
    moved = Counter()
    for i, record in enumerate(records):
        new = assignment[find(i)]
        indices[new].append(i)
        if new != record[0]:
            moved[(record[0], new)] += 1
    return indices, {
        "connected_groups": len(groups),
        "moved_rows": sum(moved.values()),
        "moves": {
            f"{old}->{new}": count for (old, new), count in sorted(moved.items())
        },
        "split_sizes": {split: sizes[split] for split in splits},
    }


def make_leakage_safe_splits(published, records, expected_sizes, seed=42):
    """Select every original row once; image data stays in the existing cache."""
    indices, report = group_assignments(records, expected_sizes, seed)
    fingerprint = sha256()
    for split, selected_indices in indices.items():
        for index in selected_indices:
            _, _, caption, image_hash, _ = records[index]
            fingerprint.update(f"{split}\0{index}\0{caption}\0{image_hash}\n".encode())
    report["split_fingerprint_sha256"] = fingerprint.hexdigest()
    combined = concatenate_datasets([published[split] for split in expected_sizes])
    repaired = DatasetDict(
        {
            split: combined.select(indices[split], keep_in_memory=True)
            for split in expected_sizes
        }
    )
    return repaired, report
