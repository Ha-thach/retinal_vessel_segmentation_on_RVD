import os
import time
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import yaml
from operator import add

from utils.utils import create_dir, seeding, load_data


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
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


if __name__ == "__main__":
    """ Load configuration from YAML """
    with open("../../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """ Seeding """
    seeding(config['seed'])

    """ Folders to save test images"""
    result_folder_path = config['result_path']
    images_result_path = os.path.join(result_folder_path, "concatenate_result_on_fractal_train_2_dataset")
    create_dir(images_result_path)
    mask_result_path = os.path.join(result_folder_path, "mask_result_on_fractal_train_2_dataset")
    create_dir(mask_result_path)
    results_csv_path = os.path.join(result_folder_path, "results.csv")
    """ Load dataset """
    """ Load dataset """
    data_path = config['data_path']  # Assuming the path is provided in config
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    data_str = f"Dataset Size:\nTest: {len(test_x)}"
    print(data_str)
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    #print("\nModel's state_dict keys and shapes (after loading checkpoint):")
    #for key, value in model.state_dict().items():
    #   print(f"{key}: {value.shape}")

    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []


    for i, (x, y) in tqdm(enumerate(zip(train_x, train_y)), total=len(train_x)):

        name = os.path.splitext(os.path.basename(x))[0]
        image = cv2.imread(x, cv2.IMREAD_COLOR)  ## (512, 512, 3)
        x = np.transpose(image, (2, 0, 1))  ## (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x).to(device)


        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (512, 512)
        y = np.expand_dims(mask, axis=0)  # (1, 512, 512)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)  # (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y).to(device)
        with torch.no_grad():

            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()  # (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)  # (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)
            """ Saving masks """
            ori_mask = mask_parse(mask)
            pred_y = mask_parse(pred_y)
            line = np.ones((size[1], 10, 3)) * 128

            cat_images = np.concatenate([image, line, ori_mask, line, pred_y * 255], axis=1)
            print(f"Saving image: {images_result_path}/{name}.png")
            cv2.imwrite(f"{images_result_path}/{name}.png", cat_images)
            """Saving masks only"""
            pred_mask_image= pred_y * 255
            print(f"Saving image: {mask_result_path}/{name}.png")
            cv2.imwrite(f"{mask_result_path}/{name}.png", pred_mask_image)

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
