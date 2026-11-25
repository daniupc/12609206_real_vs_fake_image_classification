# First checkpointed Defactify results

These are results from the pinned 96,000-image Defactify revision and the repaired split described in [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md). The three models used the same 41,742 training and 8,784 validation rows, five epochs, seed 42, 224 × 224 inputs, batch size 32, AdamW at 1e-4, and a fixed 0.5 probability cutoff. Checkpoints and detailed epoch CSVs remain local under ignored `checkpoints/` and `outputs/`. This document records aggregate results so that progress is visible in Git without distributing data or weights.

## Validation comparison

| Model | Best epoch | Balanced accuracy | AUROC |
| --- | ---: | ---: | ---: |
| RGB ResNet-18 | 3 | 0.9577 | 0.9919 |
| RGB + Fourier, concatenation | 4 | 0.9559 | 0.9915 |
| RGB + Fourier, attention + auxiliary source head | 3 | **0.9658** | **0.9942** |

The attention + auxiliary model was selected using validation balanced accuracy. Concatenation did not beat the RGB baseline. Because attention and the auxiliary head changed together, this run cannot isolate which, if either, produced the validation improvement. These are single-seed runs, not uncertainty estimates.

## One-time final test

The selected model was evaluated once on 45,474 repaired test rows: 7,579 real and 37,895 generated. `outputs/final_test_metrics.json` is the local source artifact. Its values are:

| Metric | Validation at selected epoch | Final test |
| --- | ---: | ---: |
| Balanced accuracy | 0.9658 | **0.8808** |
| Balanced error rate | 0.0342 | **0.1192** |
| AUROC | 0.9942 | 0.9734 |
| Real-photo recall | 0.9781 | 0.9790 |
| Generated-image recall | 0.9534 | **0.7827** |

Test confusion matrix (rows: true real, AI; columns: predicted real, AI): `[[7420, 159], [8236, 29659]]`. Thus 8,236 generated images were called real. AI precision was 0.9947 and AI F1 was 0.8760. Ordinary accuracy was 0.8154, below the 0.8333 obtained by always predicting the majority AI class; that trivial rule has only 0.5 balanced accuracy. The project therefore keeps balanced error as its main metric. The stated test target, balanced error ≤ 0.10, **was not met**.

| Test source | Recall | Incorrect / 7,579 |
| --- | ---: | ---: |
| Real | 0.9790 | 159 |
| SD 2.1 | **0.6939** | **2,320** |
| SDXL | 0.8344 | 1,255 |
| SD 3 | **0.7473** | **1,915** |
| DALL-E 3 | 0.8039 | 1,486 |
| Midjourney v6 | 0.8338 | 1,260 |

The validation–test balanced-accuracy gap is 0.0849. The real class is stable; the generated class causes nearly all of the recall loss. Test AUROC remains high, so threshold placement is worth investigating **on validation only**, but the lower test AUROC also means a threshold explanation alone is unproven. Neither a distribution shift nor a compression shortcut has been established as the cause.

## Next controlled experiments

1. Run the notebook's artifact-only table to inspect all epochs, the validation–test contrast, and per-source test recall without scoring test again.
2. Optionally run its **validation-only** diagnostics to collect per-source counts and study a threshold using validation logits. Keep that output exploratory; do not change the already reported test result or tune to it.
3. Before another claim of improvement, predeclare separate experiments: attention-only and auxiliary-only ablations, a common-format/size control, and a genuinely new held-out source or dataset. Compare each with the RGB baseline under the same protocol. The current test has been viewed and should not be reused as an unbiased model-selection target.

The repaired split removes measured exact-caption and byte-identical cross-split overlap, but not perceptual near-duplicates or source-dependent compression/size cues. This same-dataset test cannot establish generalization to arbitrary images or unseen generators. A score is an estimate, not proof about one image.
