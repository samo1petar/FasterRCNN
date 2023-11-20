import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_figure(image: np.ndarray, gt: np.ndarray, pred: np.ndarray, name: str, destination: str) -> None:

    plt.close('all')
    fig = plt.figure(figsize=(5, 7))
    ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3, rowspan=1)

    ax1.imshow(image)
    ax1.set_title(name, fontsize=12)

    ax2.plot(gt, label='GT')
    ax2.plot(pred, label='Prediction')
    ax2.legend()
    ax2.set_xlabel('classes', fontsize=10)
    ax2.set_ylabel('probability', fontsize=10)
    ax2.set_title('Gt & Output', fontsize=10)

    plt.tight_layout()
    plt.savefig(destination)


def save_detections(image: np.ndarray, gt: np.ndarray, predictions: np.ndarray, name: str, destination: str) -> None:

    for pred_sample in predictions.round().astype(np.int32):
        cv2.rectangle(image, (pred_sample[1], pred_sample[0]), (pred_sample[3], pred_sample[2]), (255, 0, 0), 1)

    for gt_sample in gt.round().astype(np.int32):
        cv2.rectangle(image, (gt_sample[1], gt_sample[0]), (gt_sample[3], gt_sample[2]), (0, 255, 0), 1)

    cv2.imwrite(destination, image)
