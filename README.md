# Detecting AI-generated images

Applied Deep Learning — Assignments 1 and 2

## Project

I want to train a model that can tell a real photograph from an image made by a generative model. I use the [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) and keep all 96,000 images. The task is binary classification: real or AI-generated. I am not trying to detect edits to an otherwise real photo.

This is a **Bring your own method** project. The implemented notebook trains a ResNet-18 RGB baseline, a two-branch RGB/Fourier model with simple concatenation, and a two-branch RGB/Fourier model with attention fusion plus an auxiliary source-label head. The comparison is meant to show whether the Fourier branch, attention fusion, and extra source-prediction task help under the same split and metric. The attention and auxiliary-head variant is currently tested as one combined model, so its validation result should not be read as a separate ablation of attention alone or the auxiliary head alone.

The upload frontend is intentionally deferred to Assignment 3. The Assignment 2 work focuses on the dataset audit, leakage-safe splits, model training code, validation comparison, tests for split repair, training checkpoints, CSV logging, and model-loading utilities, and reproducible instructions. The final app should present the detector score as an estimate, not proof that an image is genuine or fake.

## Related research

- [Roy et al., *A Comprehensive Dataset for Human vs. AI Generated Image Detection* (2026)](https://arxiv.org/abs/2601.00553) introduce the dataset used here. They describe real MS COCO photographs and images generated from their captions by five models, and provide a frequency-domain ResNet-50 baseline. Their paper reports a binary-task score of 0.80144, but I will not treat this as directly comparable until the metric and evaluation protocol are matched.
- [Wang et al., *CNN-Generated Images Are Surprisingly Easy to Spot... for Now* (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html) show why a pretrained CNN is a useful detector baseline and why training choices matter for transfer. They use ResNet-50, not ResNet-18; I chose the smaller ResNet-18 to make the first full-data experiment manageable, not because the paper shows it is the best architecture.
- [Corvi et al., *Intriguing Properties of Synthetic Images: From Generative Adversarial Networks to Diffusion Models* (CVPRW 2023)](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_to_CVPRW_2023_paper.html) find frequency-domain differences in generated images, including diffusion images. Together with Roy et al.'s frequency baseline, this motivates testing a Fourier branch, but does not establish that it will improve this dataset's binary task.
- [Zhu et al., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image* (NeurIPS 2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html) motivate testing both cross-generator generalization and robustness to compression or other degradations. GenImage is background research, not an additional training dataset for this project.
- [Grommelt et al., *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets* (2024)](https://arxiv.org/abs/2403.17608) document a relevant failure mode: a detector may learn image size or compression differences rather than evidence of generation. I will audit and control these factors when comparing models.
- [Yang, *FreqCross: A Multi-Modal Frequency-Spatial Fusion Network for Robust Detection of Stable Diffusion 3.5 Generated Images* (2025)](https://arxiv.org/abs/2507.02995) motivates combining RGB and Fourier features. Its published model uses feature concatenation on a different dataset; the attention fusion and auxiliary source-prediction head proposed below are this project's own adaptation, to be evaluated rather than assumed beneficial.
- [Ojha et al., *Towards Universal Fake Image Detectors That Generalize Across Generative Models* (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html) study detection with features from a pretrained CLIP image encoder, including a linear classifier, and report transfer to generators unseen during training. This motivates a frozen-feature comparison here; it does not imply CLIP will outperform the other models on Defactify.

## Data

The [Defactify dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset), also called MS COCOAI in its paper, contains 16,000 photographs from MS COCO. Each photograph has a caption. The authors used those captions to make one new image with each of five generators, giving 80,000 generated images and 96,000 images in total.

| Source | Images | Source label |
| --- | ---: | ---: |
| Real photographs | 16,000 | 0 |
| Stable Diffusion 2.1 | 16,000 | 1 |
| Stable Diffusion XL | 16,000 | 2 |
| Stable Diffusion 3 | 16,000 | 3 |
| DALL·E 3 | 16,000 | 4 |
| Midjourney v6 | 16,000 | 5 |
| **Total** | **96,000** | |

The dataset card lists 42,000 training images, 9,000 validation images, and 45,000 test images. It provides the image, caption, binary real/fake label, and source label. Hugging Face lists the files at about 7.51 GB. I download the full dataset locally, but the images and Hugging Face cache stay out of Git.

There are five times as many generated images as real ones. I keep every image and deal with the imbalance in the training loss rather than discard data.

The pre-training audit found leakage in the published split: 60 normalized-caption groups and 12 exact image-byte groups crossed train/validation/test. The project therefore uses the replacement split documented in [DATA_AUDIT_REPORT.md](DATA_AUDIT_REPORT.md). It groups records transitively by normalized caption or exact image-byte hash, keeps each group inside one split, and preserves all 96,000 images. The repaired split sizes are 41,742 train, 8,784 validation, and 45,474 test records. A second audit found zero measured caption or byte-identical overlap across the repaired splits.

## Method and tests

The notebook builds the model in stages:

1. Fine-tune a pretrained ResNet-18 on RGB images to establish a manageable CNN baseline, following the reason for a CNN comparison in Wang et al. rather than claiming their ResNet-50 results for this smaller model.
2. Add a small convolutional branch that reads the log-magnitude of a 2D Fourier transform. Corvi et al. and Roy et al. motivate looking for frequency clues. The first two-branch version joins the RGB and Fourier features by simple concatenation, as in the spatial-frequency comparison by Yang.
3. Replace that simple join with a small attention fusion layer and add a second output that predicts the source label: real or one of the five generators. Roy et al.'s dataset provides these labels, but the binary label can already be derived from them. This extra task might help, or it might make the model rely on fingerprints of known generators, so it needs cautious interpretation.

The main error metric is **balanced error rate = 1 − balanced accuracy** on the binary real/fake task, because the dataset has a 1:5 real/generated class imbalance. The notebook reports balanced accuracy alongside it, plus precision, recall, F1, AUROC, a confusion matrix, and recall by source for the final evaluation. Predicting "AI-generated" for every image would already give 83.3% ordinary accuracy while missing every real photograph, so ordinary accuracy alone is not a useful target.

The proposed target was **balanced error rate ≤ 0.10 on the repaired, held-out test split** (equivalently balanced accuracy ≥ 0.90). It was stated after seeing an earlier validation run but before the one-time final test, so it was not independently pre-registered. The observed final-test balanced error is **0.1192**, which misses the target. Neither threshold proves generalization outside Defactify.

An earlier notebook run printed these validation results without saving checkpoints or epoch CSVs. They have since been superseded by the checkpointed results in [RESULTS.md](RESULTS.md):

| Model | Best validation balanced accuracy |
| --- | ---: |
| RGB ResNet-18 baseline | 0.9482 |
| RGB/Fourier concatenation | 0.9586 |
| RGB/Fourier attention + auxiliary source head | 0.9655 |

The checkpointed attention + auxiliary model reached 0.9658 validation balanced accuracy, but only 0.8808 on the one-time test. The main concern is generated-image recall falling from 0.9534 to 0.7827. The notebook now includes artifact-only diagnostic tables and optional validation-only source/threshold checks; the viewed test must not become a tuning set. Compression/resizing controls, held-out-generator evaluation, and a frozen-CLIP comparison remain future work.

## Reproducing the project

The code is developed with Python 3.11. Dependencies are listed in both [environment.yml](environment.yml) and [requirements.txt](requirements.txt).

Fresh Conda setup:

```bash
conda env create -f environment.yml
conda activate applied
python -m ipykernel install --user --name applied --display-name "Python (applied)"
jupyter lab 01_defactify.ipynb
```

If Conda is not available, create a Python 3.11 virtual environment and install the same package set:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab 01_defactify.ipynb
```

Run the lightweight tests from the activated `applied` environment with:

```bash
python -m unittest discover -s tests -v
```

Python source files and notebook code cells were formatted with Black 24.10.0. To check or repeat formatting, install the optional development formatter (`python -m pip install 'black[jupyter]==24.10.0'`) and run:

```bash
black --check defactify_*.py tests/*.py
black --check --ipynb 01_defactify.ipynb
```

The first dataset download needs enough free disk space for the Hugging Face cache. The notebook pins Defactify revision `1c3b714c9691eb6dd11c2c1e92eab1ac29584fa0`. Use `RUN_DOWNLOAD=True` only for the first download. For normal offline reruns with an existing cache, use `RUN_LOAD_DATASET=True`, `RUN_DOWNLOAD=False`, and `RUN_AUDIT=True` so the leakage-safe split is rebuilt and checked before training.

Training should only run after the audit passes. The completed notebook now defaults to `RUN_TRAIN=False`, `RUN_EXPORT_FINAL=False`, and `RUN_FINAL_TEST=False` to avoid accidental reruns. When deliberately enabled, training prints the device, split sizes, audit summary, experiment name, epoch, losses, balanced accuracy, and best validation score. The artifact-only results cell can be run without loading data. The checkpoint and logging contract is:

| Artifact | Purpose |
| --- | --- |
| `checkpoints/<experiment>/last.pt` | Latest checkpoint for that experiment. |
| `checkpoints/<experiment>/best.pt` | Best validation checkpoint for that experiment. |
| `outputs/epochs_<experiment>.csv` | Per-epoch train/validation metrics saved with pandas. |
| `outputs/final_model.pt` | Frontend-safe checkpoint bundle for the selected model. |
| `outputs/final_weights.pt` | Selected model weights only, for code that reconstructs the architecture separately. |
| `outputs/final_model.json` | Metadata needed by the frontend, including selected experiment and image size. |

The repository ignores `data/`, `outputs/`, and `checkpoints/`, so datasets, model weights, and generated experiment outputs are not committed.

For Assignment 3, the frontend can load the exported bundle without opening the notebook or downloading pretrained weights:

```python
from defactify_models import load_detector

detector = load_detector("outputs/final_model.pt", device="cpu")
result = detector.predict("path/to/uploaded_image.jpg")
print(result["ai_probability"], result["caution"])
```

The bundle was generated after the three checkpointed training runs completed. The returned score is an estimate, not proof about an individual image.

## Assignment 2 status

| Requirement | Current project status |
| --- | --- |
| Implement the project | Notebook implements loading, audit/repair, three model variants, validation comparison, and final-test code. The final frontend is deferred to Assignment 3. |
| Metric and target | Main error metric is binary balanced error rate. Proposed final-test target is ≤ 0.10, declared after the earlier validation run; 0.96 validation balanced accuracy is a practical milestone. |
| Achieved metric | Checkpointed attention + auxiliary reached 0.9658 validation balanced accuracy and 0.8808 final-test balanced accuracy (test BER 0.1192). The ≤ 0.10 BER target was missed. See [RESULTS.md](RESULTS.md). |
| Tests | The unit tests cover split repair, model output shapes, image preprocessing for inference, checkpoint loading, binary/source metrics, validation-only threshold selection, CSV epoch logging, checkpoint resume, and final-model export. |
| Easy run | `environment.yml`, `requirements.txt`, and the commands above document setup and notebook execution. |
| Python version | Python 3.11. |
| No binaries in Git | `.gitignore` excludes PDFs, data caches, outputs, checkpoints, notebooks checkpoints, and Python caches. |

Actual programming hours were not tracked during the work already done, so I should not invent them for the report. The table below is the honest Assignment 2 status:

| WBS item | Estimated hours | Actual hours |
| --- | ---: | --- |
| Download and check the dataset; prepare images and labels | 8 | Not recorded |
| Design and code the baseline and two-branch model | 12 | Not recorded |
| Train and tune the models | 12 | Not recorded |
| Evaluate results and inspect mistakes | 8 | Not recorded |
| Build and test the upload app | 5 | Deferred to Assignment 3 |
| **Programming exercise** | **45** | **Not recorded** |
| Write the final report | 6 | Not recorded |
| Prepare the presentation | 4 | Not recorded |
| **Report and presentation preparation** | **10** | **Not recorded** |
| Present the final results | 4 | Not recorded |
| **Total** | **59** | **Not recorded** |

## Time plan

These estimates are hours of my work, not time spent waiting for a training job to finish.

| Work | Hours |
| --- | ---: |
| Download and check the dataset; prepare images and labels | 8 |
| Design and code the baseline and two-branch model | 12 |
| Train and tune the models | 12 |
| Evaluate results and inspect mistakes | 8 |
| Build and test the upload app | 5 |
| **Programming exercise** | **45** |
| Write the final report | 6 |
| Prepare the presentation | 4 |
| **Report and presentation preparation** | **10** |
| Present the final results | 4 |
| **Total** | **59** |

## References

- Roy, R., et al. (2026). [*A Comprehensive Dataset for Human vs. AI Generated Image Detection*](https://arxiv.org/abs/2601.00553). arXiv:2601.00553.
- Wang, S.-Y., et al. (2020). [*CNN-Generated Images Are Surprisingly Easy to Spot... for Now*](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.html). CVPR.
- Corvi, R., et al. (2023). [*Intriguing Properties of Synthetic Images: From Generative Adversarial Networks to Diffusion Models*](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_to_CVPRW_2023_paper.html). CVPR Workshops.
- Zhu, M., et al. (2023). [*GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*](https://papers.neurips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html). NeurIPS 36, Datasets and Benchmarks Track.
- Grommelt, P., et al. (2024). [*Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*](https://arxiv.org/abs/2403.17608). arXiv:2403.17608.
- Yang, G. (2025). [*FreqCross: A Multi-Modal Frequency-Spatial Fusion Network for Robust Detection of Stable Diffusion 3.5 Generated Images*](https://arxiv.org/abs/2507.02995). arXiv:2507.02995.
- Ojha, U., et al. (2023). [*Towards Universal Fake Image Detectors That Generalize Across Generative Models*](https://openaccess.thecvf.com/content/CVPR2023/html/Ojha_Towards_Universal_Fake_Image_Detectors_That_Generalize_Across_Generative_Models_CVPR_2023_paper.html). CVPR.
- [Defactify Image Dataset on Hugging Face](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset).
