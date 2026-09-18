# ToBRFV classification with paired RGB and VNIR images

Code and saved results accompanying **Swin-based fusion of paired RGB and VNIR images for ToBRFV classification in tomato and pepper**.

**Authors:**
- Mansur BEŞTAŞ, Department of Business Administration, Bitlis Eren University, Türkiye. Contact: mbestas@beu.edu.tr. ORCID: https://orcid.org/0000-0002-8192-2044.
- Ruhi TAŞ, OSTİM Technical University, Türkiye. Contact: ruhi.tas@ostimteknik.edu.tr. ORCID: https://orcid.org/0000-0002-7741-7715.

## What this is

ToBRFV (Tomato brown rugose fruit virus) is diagnosed here from four matched images of the same plant, day and viewing angle: one RGB image and three VNIR images (visible, 800 nm, 1000 nm). The proposed model adapts the final two Swin Tiny blocks with cross-entropy (CE), then learns a gated fusion head with a separate RGB projection and a shared projection for the three VNIR views. One matched plant/day/viewing-angle quartet produces one four-class prediction (Tomato_Healthy, Tomato_Virus, Pepper_Healthy, Pepper_Virus).

## Reported experiment

| Split | Plants | Complete quartets | Images |
|---|---:|---:|---:|
| Training | 244 | 7,130 | 28,520 |
| Validation | 53 | 1,559 | 6,236 |
| Test | 50 | 1,489 | 5,956 |

The proposed model correctly classified 1,412 of 1,489 test quartets: **94.828744% accuracy** and **94.545866% macro F1**. Saved predictions for the proposed model and two component baselines are included, along with the full historical candidate-selection record. See [repository/README.md](repository/README.md) for the complete methodology, reproducibility scope and limitations.

## Repository layout

```
├── repository/          # Code repository root — publish this as the GitHub project
│   ├── README.md         # Full technical README (install, reproduce, predict, retrain)
│   ├── CITATION.cff       # Citation metadata
│   ├── requirements.txt   # Python dependencies
│   ├── data/              # Paired image manifest (no raw images)
│   ├── docs/              # Provenance and validation records
│   ├── figures/           # Manuscript figures (workflow, confusion/ROC/PR)
│   ├── results/           # Saved predictions, training histories, metrics
│   ├── scripts/           # train.py, evaluate.py, predict.py, model.py, plotting
│   └── weights/           # Empty here — see "Model weights" below
├── release_assets/       # Local copies of the three .pt checkpoints (not pushed to GitHub)
├── MANIFEST_SHA256.json  # SHA-256 checksums for every file in this submission package
├── YUKLEME_TR.md         # Upload/packaging notes for the author (Turkish)
└── UPLOAD_EN.md          # Same upload/packaging notes (English)
```

## Quick start

```bash
cd repository
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -r requirements.txt

# Recompute reported metrics/figures from saved predictions (no images or weights needed)
python scripts/evaluate.py
python scripts/plot_results.py
python scripts/plot_workflow.py
```

Full install, prediction/Grad-CAM and retraining instructions are in [repository/README.md](repository/README.md).

## Dataset and model weights

The image dataset (TOBRFV-LMID) is not redistributed here; obtain it from https://doi.org/10.5281/zenodo.17244968 (CC BY 4.0) and cite its data paper, https://doi.org/10.1016/j.dib.2026.112927. `repository/data/paired_manifest.csv` lists the exact matched observations used in the reported experiment.

The three trained checkpoints (`base_swin.pt`, `ce_tail.pt`, `ce_gated.pt`) exceed GitHub's upload limits and are hosted on Google Drive as a single zip archive:

https://drive.google.com/file/d/15mHnY6mqvAymrpn6LghK3quhytknkBJX/view?usp=drive_link

Download and extract the archive, then place the three `.pt` files in `repository/weights/`. SHA-256 checksums are in `repository/weights/checksums.json`. `base_swin.pt` is already trained on ToBRFV — replacing it with an ImageNet model does **not** reproduce the experiment.

## Citation and reuse

Cite the manuscript, the TOBRFV-LMID dataset and the original Swin Transformer / Grad-CAM methods when using this repository. See [repository/CITATION.cff](repository/CITATION.cff) for author metadata. No software reuse license has been selected yet: public availability does not by itself grant an open-source license, and the dataset, third-party libraries and pretrained weights retain their own terms.
