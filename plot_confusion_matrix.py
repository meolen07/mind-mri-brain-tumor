from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from src.inference import load_model, predict

FOLDER_TO_IDX = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'no_tumor': 2,
    'pituitary': 3,
}

DISPLAY_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
PRED_NAME_TO_IDX = {
    'Glioma': 0,
    'Meningioma': 1,
    'No Tumor': 2,
    'Pituitary Tumor': 3,
}


def _collect_testing_samples(testing_dir: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for class_dir in sorted(testing_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        key = class_dir.name.lower().replace(' ', '_')
        if key not in FOLDER_TO_IDX:
            continue
        label = FOLDER_TO_IDX[key]
        for path in class_dir.rglob('*'):
            if path.is_file() and path.suffix.lower() in EXTS:
                samples.append((path, label))
    return samples


def main() -> None:
    root = Path(__file__).resolve().parent
    testing_dir = root / 'data' / 'Testing'
    weights_path = root / 'weights' / 'best_model.pth'

    model = load_model(weights_path=weights_path, device='cpu')
    samples = _collect_testing_samples(testing_dir)
    if not samples:
        raise RuntimeError(f'No testing images found under {testing_dir}')

    y_true: list[int] = []
    y_pred: list[int] = []

    for image_path, label in samples:
        with Image.open(image_path) as img:
            pred_name, _conf, _probs = predict(model, img, device='cpu')
        y_true.append(label)
        y_pred.append(PRED_NAME_TO_IDX[pred_name])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=140)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title('MIND Confusion Matrix on Testing Set')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks(np.arange(len(DISPLAY_NAMES)))
    ax.set_yticks(np.arange(len(DISPLAY_NAMES)))
    ax.set_xticklabels(DISPLAY_NAMES, rotation=20, ha='right')
    ax.set_yticklabels(DISPLAY_NAMES)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()

    out_path = root / 'report' / 'figures' / 'confusion_matrix.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(out_path)


if __name__ == '__main__':
    main()
