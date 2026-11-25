# Defactify data audit and split repair

This note records a data-quality finding for the final report. It is about the dataset and evaluation protocol, **not** a model result. The audit ran locally on all 96,000 records of Defactify revision `1c3b714c9691eb6dd11c2c1e92eab1ac29584fa0`, using seed `42` for the replacement split. Dataset images are not included here.

## What the published splits contain

| Check | Published train/validation/test |
| --- | ---: |
| Records | 42,000 / 9,000 / 45,000 |
| Invalid labels or unreadable images | 0 |
| Normalized-caption groups crossing splits | 60 (1,758 affected records) |
| Exact image-byte hash groups crossing splits | 12 (54 affected records) |

The 12 cross-split identical-image groups are all labelled Midjourney v6. In the inspected groups, identical image bytes appeared under different captions, so grouping by caption alone would not have removed this leakage. A repeated caption does not necessarily mean the underlying photograph is identical; the byte-identical images are unambiguous overlap. The original split should therefore not be treated as an independent held-out test for this project.

All six sources contain 16,000 records. All files inspected by the audit are JPEG. The real source has 842 distinct image dimensions, while the five generated sources have only one or two each. This is a potential size or processing shortcut for a detector, not evidence that a trained model uses it.

## Repair used for experiments

The notebook first audits the published data. If it finds leakage, `defactify_splits.py` joins records into connected groups whenever they share either a whitespace/case-normalized caption or a SHA-256 hash of the exact image bytes. It assigns each complete group to the split containing most of its original rows; ties use the seeded order and current size deficit. It then audits the replacement split before constructing training loaders. No original record is removed or copied. The data remain in the existing local Hugging Face cache; the replacement uses in-memory row indices, not a second image download.

| Split | Published records | Replacement records | Records per source in replacement |
| --- | ---: | ---: | ---: |
| Train | 42,000 | 41,742 | 6,957 |
| Validation | 9,000 | 8,784 | 1,464 |
| Test | 45,000 | 45,474 | 7,579 |
| **Total** | **96,000** | **96,000** | **16,000 per source overall** |

The algorithm moved 642 records: 402 train→test, 12 train→validation, 72 validation→test, and 156 validation→train. It formed 9,826 connected caption/hash groups. The changed split sizes are close to the published proportions but not identical because linked groups are indivisible; retaining every image and eliminating measured leakage took priority over exact split counts.

The full audit of the replacement found **zero** cross-split normalized-caption groups, **zero** cross-split byte-identical image groups, and **zero** invalid labels or unreadable images. Each replacement split still has equal counts for all six sources, preserving the 1:5 real/AI class ratio. No model training or test-set evaluation is claimed in this note.

## Wording for the final report

> We audited all 96,000 images in the pinned Defactify revision before training. The published splits contained 60 captions appearing in multiple splits and 12 groups of byte-identical images crossing splits. To avoid this measured leakage, we grouped records transitively by normalized caption or exact image-byte hash and assigned each complete group to one split using a fixed seed. This retained all images and produced train, validation and test sets of 41,742, 8,784 and 45,474 records. A second full audit found no cross-split overlaps under either check. We selected models using the replacement validation set and reserved its test set for final evaluation. Results on these replacement splits are not directly comparable with scores reported on the published splits.

Use the last two sentences only after the corresponding training and final evaluation have actually happened.

## Remaining limits

- Normalized-caption equality can group different scenes with common text; it is a conservative leakage control, not proof that the images are the same.
- Exact byte hashes miss resized, recompressed or otherwise visually near-duplicate images. A perceptual-duplicate check would provide stronger evidence.
- Multiple records can share a caption within one split (many test caption groups have 30 records). Row-level metrics may therefore give more weight to these repeated caption groups; group-aware uncertainty estimates would be preferable.
- Equal JPEG format does not equalize compression settings or image dimensions. The large dimension difference by source needs scrutiny when interpreting model performance; resizing to 224×224 alone does not rule out shortcuts.
- The new test split is held out from model selection, but it comes from the same dataset and generators. It cannot establish generalization to arbitrary new generators or real-world images.
- The dataset card did not provide an explicit redistribution license when this project was prepared; do not publish image files without confirming rights.
