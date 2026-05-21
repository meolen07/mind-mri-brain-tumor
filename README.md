# MIND — MRI-based Intelligent Neural Detection

**Deep learning brain MRI classifier** (Glioma, Meningioma, No Tumor, Pituitary Tumor) with an EfficientNet-B0 backbone and a Streamlit demo for educational and research use.

| | |
|---|---|
| **Live demo** | [https://mind-2026.streamlit.app/](https://mind-2026.streamlit.app/) |
| **Report** | [https://github.com/meolen07/mind-mri-brain-tumor/blob/main/report.pdf](https://github.com/meolen07/mind-mri-brain-tumor/blob/main/report.pdf) |
| **Repository** | [https://github.com/meolen07/mind-mri-brain-tumor](https://github.com/meolen07/mind-mri-brain-tumor) |

## Overview

MIND is an end-to-end pipeline for four-class brain tumor MRI classification: download the public Kaggle dataset, fine-tune EfficientNet-B0 in PyTorch, and run interactive inference in the browser. The app reports the predicted class, confidence, and per-class probabilities. It is intended for **learning and research**—not clinical diagnosis.

## Classes

| Index | Class name | Kaggle folder (`data/Training/`) |
|------:|------------|----------------------------------|
| 0 | Glioma | `glioma` |
| 1 | Meningioma | `meningioma` |
| 2 | No Tumor | `notumor` or `no_tumor` |
| 3 | Pituitary Tumor | `pituitary` |

## Features

- **Streamlit web UI** — upload brain MRI images (JPG, JPEG, PNG)
- **EfficientNet-B0** — ImageNet-pretrained backbone with a four-class head; CPU inference by default
- **Probability outputs** — predicted label, confidence, and sorted bar chart of class probabilities
- **Cached model loading** — `st.cache_resource` for faster repeat predictions
- **Training script** — stratified train/validation split; checkpoint saved to `weights/best_model.pth`
- **Optional plots** — `plot_confusion_matrix.py` and `plot_training_curve.py` regenerate report figures

## Results

| Metric | Value |
|--------|--------|
| Best validation accuracy | **99.52%** (epoch 8) |
| Final epoch validation accuracy | 99.05% |
| Hardware | CPU |
| Split | 4,760 train / 840 validation images (85/15 stratified) |
| Training | 15 epochs, Adam, EfficientNet-B0 fine-tuning |

See the **[Report](https://github.com/meolen07/mind-mri-brain-tumor/blob/main/report.pdf)** PDF for the confusion matrix, training curves, and full write-up.

## Project structure

```
MIND/
├── app.py                      # Streamlit demo
├── train.py                    # Training entry point
├── requirements.txt
├── plot_confusion_matrix.py    # Regenerate confusion matrix figure
├── plot_training_curve.py      # Regenerate training curve figure
├── report.pdf                  # Project report (PDF)
├── src/
│   ├── model.py                # EfficientNet-B0 builder
│   ├── inference.py            # Preprocess + predict
│   └── gradcam.py              # Grad-CAM placeholder (future)
├── weights/
│   └── best_model.pth          # Trained checkpoint (required for inference)
└── data/                       # Not in repo — download from Kaggle (see below)
    └── Training/
        ├── glioma/
        ├── meningioma/
        ├── notumor/
        └── pituitary/
```

## Dataset

Images come from the public **[Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)** dataset on Kaggle (author: masoudnickparvar / nickparvar).

The `data/` directory is **not** committed to this repository (listed in `.gitignore` because of size). After download, extract so class folders live under:

```
data/Training/glioma/
data/Training/meningioma/
data/Training/notumor/   
data/Training/pituitary/
```

`data/Testing/` is optional; `train.py` uses `data/Training/` only.

## Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/meolen07/mind-mri-brain-tumor.git
cd mind-mri-brain-tumor
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add data and train (or use bundled weights)

Download the Kaggle dataset into `data/` as above, then:

```bash
python train.py --data_dir data --epochs 15
```

Checkpoints are written to `weights/best_model.pth` (format expected by `src/inference.py`). For a quick demo, use the checkpoint already in the repo if present.

### 3. Run the app locally

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Deployment

The public demo is hosted on **[Streamlit Community Cloud](https://mind-2026.streamlit.app/)**:

1. Connect the [GitHub repository](https://github.com/meolen07/mind-mri-brain-tumor).
2. Set the main file to `app.py`.
3. Ensure `requirements.txt` and `weights/best_model.pth` are present in the repo layout.

## Medical disclaimer

This software is for **educational and research purposes only**. It is **not** intended to diagnose, treat, or prevent any disease and must **not** replace professional medical judgment. Always consult qualified healthcare professionals for medical decisions.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for the full text.

## Citation

If you use MIND in academic or research work, please cite the repository, the dataset, and EfficientNet as appropriate:

```bibtex
@misc{mind_mri_brain_tumor_2026,
  author       = {Nguyen, Huynh Mai Linh},
  title        = {MIND: MRI-based Intelligent Neural Detection},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/meolen07/mind-mri-brain-tumor}
}

@misc{nickparvar_brain_tumor_mri,
  author       = {Nickparvar, Masoud},
  title        = {Brain Tumor MRI Dataset},
  howpublished = {Kaggle},
  year         = {2023},
  url          = {https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}
}

@inproceedings{tan2019efficientnet,
  author    = {Tan, Mingxing and Le, Quoc V.},
  title     = {EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year      = {2019},
  pages     = {6105--6114},
  publisher = {PMLR},
  note      = {arXiv:1905.11946}
}
```

## Author

**Huynh Mai Linh Nguyen** — 2026
