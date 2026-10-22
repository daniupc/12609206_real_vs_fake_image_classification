# Detecting AI-generated images

Applied Deep Learning — Assignment 1: Initiate

## Project

I want to train a model that can tell a real photograph from an image made by a generative model. I will use the [Defactify Image Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) and work with all 96,000 images. The task is binary classification: real or AI-generated. I am not trying to detect edits to an otherwise real photo.

This is a **Bring your own method** project. My model will look at the image in two ways. One part will process the RGB pixels; another will process a Fourier transform of the image, which shows its frequency patterns. I will first combine the two parts simply, then test whether attention improves the combination. I will also test whether asking the model to identify the image source during training helps the main real/fake decision.

I will train a standard ResNet-18 classifier on the same data as a baseline. The point of the comparison is to find out whether the second branch, attention layer, and extra training task actually help. If time allows, I will also compare a frozen CLIP image encoder with a small binary classifier trained on its features. This is a separate baseline, not another branch of my model. I will build a small app where someone can upload an image and see the model's score. The score will be presented as an estimate, not proof that an image is genuine or fake.

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

The dataset card lists 42,000 training images, 9,000 validation images, and 45,000 test images. It provides the image, caption, binary real/fake label, and source label. Hugging Face lists the files at about 7.51 GB. I will download the full dataset, but not put the images in the Git repository.

There are five times as many generated images as real ones. I will keep every image and deal with the imbalance in the training loss rather than discard data. Before training, I will check the labels, image formats, duplicates, and splits. In particular, images made from the same caption should not end up in both training and test data. If the published splits have that problem, I will make new splits by caption and document the change. I will not redistribute the images unless their license allows it.

## Method and tests

I plan to build the model in stages:

1. Fine-tune a pretrained ResNet-18 on RGB images to establish a manageable CNN baseline, following the reason for a CNN comparison in Wang et al. rather than claiming their ResNet-50 results for this smaller model.
2. Add a small convolutional branch that reads the log-magnitude of a 2D Fourier transform. Corvi et al. and Roy et al. motivate looking for frequency clues. I will first join the branches by simple concatenation, as in the spatial-frequency comparison by Yang, so I can measure what the frequency branch adds.
3. Replace that simple join with a small cross-attention layer, working on reduced-size feature maps to limit memory use. This is my proposed way to let the branches interact, not a benefit established by the cited papers; I will keep it only if it improves the simple fusion under the same evaluation protocol.
4. Add a second output that predicts the source label: real or one of the five generators. Roy et al.'s dataset provides these labels, but the binary label can already be derived from them. This extra task might help or might make the model rely on fingerprints of known generators, so I will test it separately, including on a generator left out of training.

If time allows, I will also train a linear real/fake classifier on features from a frozen CLIP image encoder, following the comparison motivated by Ojha et al. Only the classifier will be trained on Defactify; CLIP's weights will stay fixed. This tests whether general pretrained image features transfer better than the detectors trained for this dataset, without treating CLIP as a guaranteed winner.

I will use the validation set to choose settings and keep the test set for final results. I will report a confusion matrix, balanced accuracy, precision, recall, F1, and AUROC. I will also show results for each generator. To check how fragile the model is, I will test compressed and resized images and run one experiment with a generator left out of training. In that experiment, I will evaluate the real/fake output only. Across the project I will use the full dataset, although the held-out generator is necessarily absent from that particular training run.

The comparisons will show what each part of the model changes. A more complex model may do worse, especially on a new generator, and that would still be a useful result. Because five out of six images are generated, predicting "AI-generated" for every image would already give 83.3% overall accuracy while missing every real photograph; I will not use accuracy alone to judge a model. Given the time available, the full-data RGB baseline and one finished two-branch model come first. The frozen-CLIP comparison, attention, and auxiliary-head ablations follow if time permits.

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
