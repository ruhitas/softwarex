# ToBRFV classification with paired RGB and VNIR images

Code accompanying **Swin-based fusion of paired RGB and VNIR images for ToBRFV classification in tomato and pepper**.

Authors: Mansur BEŞTAŞ, Department of Business Administration, Bitlis Eren University, Türkiye. Contact: mbestas@beu.edu.tr. ORCID: https://orcid.org/0000-0002-8192-2044. Ruhi TAŞ, OSTİM Technical University, Türkiye. Contact: ruhi.tas@ostimteknik.edu.tr. ORCID: https://orcid.org/0000-0002-7741-7715.

The proposed model adapts the final two Swin Tiny blocks with cross-entropy (CE), then learns a gated fusion head. It uses a separate RGB projection and a shared projection for three VNIR views. One matched plant/day/viewing-angle quartet produces one four-class prediction. The historical checkpoint retains an unused auxiliary infection head for state-dictionary compatibility; this head is frozen and its output is not used by the proposed method.

## Reported experiment

| Split | Plants | Complete quartets | Images |
|---|---:|---:|---:|
| Training | 244 | 7,130 | 28,520 |
| Validation | 53 | 1,559 | 6,236 |
| Test | 50 | 1,489 | 5,956 |

Four incomplete training images were excluded. The proposed model correctly classified 1,412 of 1,489 test quartets: **94.828744% accuracy** and **94.545866% macro F1**. This repository includes saved predictions for the proposed model and two component baselines. `results/original_selection.json` retains the complete historical candidate-selection record, including exploratory candidates not reproduced by the focused training command.

## Install

Use Python 3.11 and a virtual environment:

```bash
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -r requirements.txt
```

The original experiment used PyTorch 2.6.0+cu124, timm 1.0.20 and an NVIDIA RTX 4060 Laptop GPU. Install the matching CUDA-enabled PyTorch build for training. Saved-prediction evaluation runs on CPU. Package versions other than torch/timm in requirements are compatibility ranges, not a claim that every combination was tested.

## Recompute reported results without images or weights

```bash
python scripts/evaluate.py
python scripts/plot_results.py
python scripts/plot_workflow.py
```

These commands verify plant-disjoint splits, probabilities and reported metrics; reproduce 5,000 plant-cluster bootstrap replicates (seed 20260909); export the split/class/modality counts; and generate the confusion matrix, ROC and precision–recall figure. Outputs go to `runs/`, preserving the archived results.

## Dataset and model files

Obtain TOBRFV-LMID from https://doi.org/10.5281/zenodo.17244968 and cite its data paper, https://doi.org/10.1016/j.dib.2026.112927. Images are not redistributed here. The source dataset is provided under CC BY 4.0; follow its attribution requirements. This work is an independent secondary analysis, with no affiliation to the data-producing team.

`data/paired_manifest.csv` contains the exact 10,178 matched observations used in the reported experiment, in their original row order. Paths are relative to a dataset directory containing `RGB`, `VNIR`, `VNIR_800nm` and `VNIR_1000nm`. Label order is Tomato_Healthy, Tomato_Virus, Pepper_Healthy, Pepper_Virus. Plant identity is the composite `group` field, not the pot number alone. All dates and viewing angles of a plant remain in one split. The manifest covers the analyzed local archive; do not infer that every other dataset revision has identical file counts or filenames.

Place these files in `weights/` (SHA-256 checksums are in `weights/checksums.json`):

- `base_swin.pt`: the original four-class, validation-selected trial_006 checkpoint.
- `ce_tail.pt`: the adapted final two blocks and LayerNorm.
- `ce_gated.pt`: the selected gated fusion head.

These three files exceed GitHub's upload limits and are hosted on Google Drive instead of a GitHub Release, as a single zip archive containing all three `.pt` files: https://drive.google.com/file/d/15mHnY6mqvAymrpn6LghK3quhytknkBJX/view?usp=drive_link. Download and extract the archive, then place the three `.pt` files directly in this repository's `weights/` folder. `base_swin.pt` is already trained on ToBRFV: replacing it with an ImageNet model does **not** reproduce the experiment. The repository starts from this checkpoint and does not recreate its earlier hyperparameter search. Only load checkpoints from trusted sources; the original base checkpoint uses PyTorch's full checkpoint format.

## Predict and produce Grad-CAM

```bash
python scripts/predict.py --rgb path/to/rgb.tiff --vnir path/to/vnir.png --nm800 path/to/800.png --nm1000 path/to/1000.png --gradcam runs/example
```

The four inputs must be from the same plant, day and viewing angle. CPU and CUDA inference are supported. Grad-CAM targets the predicted class logit of the **joint model** at `tail.blocks[-1].norm1`. Maps use spatially averaged gradients, ReLU and independent min–max normalization per modality. A zero map is preserved. Colors across modalities are not calibrated measures of absolute contribution. This is not a validated diagnostic test.

## Retrain the proposed method and component baselines

```bash
python scripts/train.py --data-root /path/to/TOBRFV-LMID --output runs/retrain
```

CUDA, raw images and `base_swin.pt` are required. The float32 activation cache alone occupies approximately 6.1 GB of disk space; allow additional space for weights and output. The script uses the archived split, Resize(255)/CenterCrop(224), ImageNet normalization, five fixed CE epochs, and the original validation-based head selection. It trains frozen-backbone gated fusion, CE plus a linear head, and CE plus gated fusion. It selects among these three before computing test results. This focused command does not rerun the two exploratory SupCon candidates retained in the historical selection record. Numerical differences across hardware/library versions are possible. The original training was not rerun during repository preparation.

## Scope of reproducibility

Saved-prediction metrics can be reproduced independently of the images and weights. Inference requires all three provided checkpoints and valid input images. Retraining starts from the supplied historical base checkpoint. Head predictions were checked against the archived test-feature tensor during packaging; source compilation, model loading and gradient-flow smoke checks are reported in `docs/validation.json`. The full raw-image pipeline was not rerun during packaging.

The test split had been examined during earlier development and is not a new external validation set. Only one plant split and one training seed are reported for the proposed method. Healthy and infected plants were grown in separate greenhouses, so plant separation alone does not exclude environmental confounding. Bootstrap intervals quantify uncertainty conditional on these trained models and this test population. The paired interval for CE versus frozen gated fusion includes zero; statistical superiority is not established.

## Attribution and reuse

The backbone is implemented through timm and uses the Swin Transformer architecture. Gated fusion follows the general family of gated multimodal methods; Grad-CAM follows Selvaraju et al. Cite the manuscript, dataset and original methods when using this repository. See `CITATION.cff` for author metadata. The manuscript has no assigned publication DOI in this package.

No software redistribution license has been selected by the author for this release. Public availability does not by itself grant an open-source license. Dataset, third-party libraries and any pretrained weights retain their respective terms. This repository does not redistribute the manuscript, the EPPO map, third-party PDFs or raw plant images.
