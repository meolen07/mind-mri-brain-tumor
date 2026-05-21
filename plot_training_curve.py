from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    train_acc = [0.8397, 0.9405, 0.9704, 0.9819, 0.9824, 0.9878, 0.9903, 0.9920, 0.9943, 0.9922, 0.9939, 0.9939, 0.9941, 0.9975, 0.9971]
    val_acc = [0.9536, 0.9774, 0.9810, 0.9845, 0.9821, 0.9857, 0.9881, 0.9952, 0.9905, 0.9905, 0.9893, 0.9917, 0.9917, 0.9881, 0.9905]
    train_loss = [0.5121, 0.1749, 0.0887, 0.0619, 0.0543, 0.0374, 0.0304, 0.0234, 0.0209, 0.0242, 0.0218, 0.0193, 0.0182, 0.0096, 0.0099]
    val_loss = [0.1638, 0.0718, 0.0499, 0.0408, 0.0593, 0.0344, 0.0422, 0.0211, 0.0313, 0.0296, 0.0332, 0.0311, 0.0340, 0.0519, 0.0337]

    epochs = list(range(1, len(train_acc) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)

    axes[0].plot(epochs, train_acc, marker='o', linewidth=2, label='Train Accuracy')
    axes[0].plot(epochs, val_acc, marker='s', linewidth=2, label='Validation Accuracy')
    axes[0].set_title('Accuracy vs. Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(epochs)
    axes[0].legend(loc='lower right')

    axes[1].plot(epochs, train_loss, marker='o', linewidth=2, label='Train Loss')
    axes[1].plot(epochs, val_loss, marker='s', linewidth=2, label='Validation Loss')
    axes[1].set_title('Loss vs. Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(epochs)
    axes[1].legend(loc='upper right')

    fig.suptitle('MIND Training Curves (Final Logged Run)')
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / 'report' / 'figures' / 'training_curve.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(out_path)


if __name__ == '__main__':
    main()
