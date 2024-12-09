import os
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import yaml
from operator import add

from utils.utils import create_dir, seeding


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  ## (H, W, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (H, W, 3)
    return mask


if __name__ == "__main__":
    """ Load configuration from YAML """
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """ Seeding """
    seeding(config['seed'])

    """ Folders to save test masks"""
    result_folder_path = config['result_path']
    masks_result_path = os.path.join(result_folder_path, "mask_results")
    create_dir(masks_result_path)
    results_csv_path = os.path.join(result_folder_path, "results.csv")

    """ Load dataset """
    test_x = sorted(glob("/data/nhthach/project/Fractal-dim/project/segmentation/images/train"))
    test_y = sorted(glob("/data/nhthach/project/Fractal-dim/project/segmentation/images/validation"))

    """ Hyperparameters """
    H = config['image_height']
    W = config['image_width']
    size = (W, H)
    model_name = config['model']
    dropout_rate = config['dropout_rate']

    """ Load the checkpoint """
    checkpoint_path = os.path.join(result_folder_path, "checkpoint.pth")

    """ Check device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    """ Import the correct model """
    if model_name == "UNet":
        from model.UNET.unet import UNet

        model = UNet(dropout_rate=dropout_rate)  # Pass dropout rate to the model
    elif model_name == "R2AttUnet":
        from model.UNET.UNET_recurrent_model import R2AttUnet

        model = R2AttUnet(dropout_rate=dropout_rate)  # Ensure this model also takes dropout_rate
    else:
        raise ValueError(f"Cannot import model: {model_name}")

    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = os.path.splitext(os.path.basename(x))[0]

        """ Reading and checking image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {x}")
            continue

        image = cv2.resize(image, size)  ## (H, W, 3)
        x = np.transpose(image, (2, 0, 1))  ## (3, H, W)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  ## (1, 3, H, W)
        x = x.astype(np.float32)
        x = torch.from_numpy(x).to(device)

        """ Reading and checking mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {y}")
            continue

        mask = cv2.resize(mask, size)  ## (H, W)
        y = np.expand_dims(mask, axis=0)  ## (1, H, W)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)  ## (1, 1, H, W)
        y = y.astype(np.float32)
        y = torch.from_numpy(y).to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()  ## (1, H, W)
            pred_y = np.squeeze(pred_y, axis=0)  ## (H, W)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        pred_y = mask_parse(pred_y)
        print(f"Saving mask: {masks_result_path}/{name}.png")
        cv2.imwrite(f"{masks_result_path}/{name}.png", pred_y * 255)

    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}")

    fps = 1 / np.mean(time_taken)
    print("FPS: ", fps)

    """ Save the evaluation metrics to a CSV file with a new section """
    with open(results_csv_path, 'a') as f:
        f.write("\n\n### New Evaluation Section ###\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Jaccard: {jaccard:1.4f}\n")
        f.write(f"F1: {f1:1.4f}\n")
        f.write(f"Recall: {recall:1.4f}\n")
        f.write(f"Precision: {precision:1.4f}\n")
        f.write(f"Accuracy: {acc:1.4f}\n")
        f.write(f"FPS: {fps:1.4f}\n")