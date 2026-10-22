# Project context

Updated: 22 October 2026

## Current state

The repository currently contains the Assignment 1 proposal in [README.md](README.md). No dataset has been downloaded here, and no training code, model, app, or experimental results exist yet. The local assignment PDF is ignored by `.gitignore`.

The user chose **AI-generated image detection** as the project from the outset of the written proposal. The target is binary classification of photographs versus fully generated images. The course project type is **Bring your own method**.

## Data and method

- Dataset: [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset), called MS COCOAI in the [paper](https://arxiv.org/abs/2601.00553).
- Published size: 96,000 images, approximately 7.51 GB on Hugging Face. There are 16,000 real photographs and 16,000 images from each of five generators. Published splits: 42,000 train, 9,000 validation, 45,000 test.
- The user wants **all 96,000 images** used across the project; do not switch to a subset merely for convenience.
- Baseline: pretrained ResNet-18 on RGB images. Planned main model: RGB and Fourier-spectrum branches, a small cross-attention fusion layer, and binary plus six-class source-label outputs.
- Main checks: caption-level split leakage, class imbalance, matching image processing across classes, per-generator performance, one held-out-generator run, and compression/resizing tests.

These are intentions in the README, not verified properties of downloaded files or completed experiments. In particular, dataset contents, license terms, local storage needs, and available compute still need to be checked.

## Time and submission

The agreed estimate is **45 hours programming + 10 hours final report and presentation preparation + 4 hours presenting results = 59 hours**. The deadline for Assignment 1 in the PDF is **27 October 2026**. Assignment 1 asks for a topic, project type, at least two scientific papers, a dataset description, and a work breakdown with hour estimates; it does not require a trained model yet.

The assignment also asks for a repository named with the student's matriculation number and a link emailed to the course address. The matriculation number has not been provided. There is no remote configured in this local repository, and no submission email has been sent.

## Next steps

1. Review and submit the README for Assignment 1 after filling any required personal/repository details.
2. Before implementation, check available disk space and GPU/CPU resources, then verify dataset access and license.
3. Download and audit the complete dataset without adding it to Git.
4. Implement the baseline first, then the planned two-branch model and tests.

Use the README for the latest user-facing scope. Update this context file when actual implementation or submission status changes.
