# Repository guidance

This repository is for the Applied Deep Learning course project. Read [README.md](README.md) for the current proposal and [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the handoff status before making changes.

## Scope

- The task is to detect fully AI-generated images versus real photographs. Do not silently expand it to video, face swaps, or local image edits.
- Use the complete 96,000-image Defactify dataset. Keep the published train/validation/test roles unless a documented leakage check shows they must be replaced.
- The planned method is a ResNet-18 RGB baseline and a two-branch RGB/Fourier model with attention fusion and an auxiliary source-label head. Treat these as plans until code and experiments exist.
- The current workload estimate is 45 hours of programming, 10 hours for the report and presentation preparation, and 4 hours to present the results.

## Working with data and results

- Do not commit dataset files, caches, model weights, or generated outputs without a specific reason and permission to redistribute them. Before downloading data, extend `.gitignore` for the actual local paths.
- Check labels, image formats, duplicates, class balance, and caption-level leakage before training. Keep related images in the same split if resplitting is needed.
- Fit preprocessing and tune hyperparameters on training/validation data only. Keep the test set out of model selection.
- Compare models under the same data and evaluation protocol. Report negative results and the limits of any claimed generalization.
- Do not present a detector score as proof that an arbitrary image is genuine or fake.

## Editing and verification

- Keep the README as a natural, student-written proposal in English. Preserve its factual related-research section unless the user requests a change.
- Prefer small, reproducible changes and record the seed, configuration, data split, and metric definitions once implementation starts.
- Verify changed files and claims before reporting completion. Do not describe planned work as already implemented.
