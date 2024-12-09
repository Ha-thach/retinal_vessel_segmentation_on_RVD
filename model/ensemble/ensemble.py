import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import yaml
from operator import add

from utils import create_dir, seeding, load_data


def calculate_metrics(y_true, y_pred):
    """ Calculate evaluation metrics. """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8).reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def apply_or_operation(pred_unet, pred_r2attunet):
    """ Apply OR operation on the predictions of two models. """
    combined_pred = torch.logical_or(pred_unet > 0.5, pred_r2attunet > 0.5)
    return combined_pred.float()


def add_label(image, label, position=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2):
    """ Add a label to the image. """
    font = cv2.FONT_HERSHEY_SIMPLEX
    labeled_image = image.copy()
    cv2.putText(labeled_image, label, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return labeled_image


if __name__ == "__main__":
    """ Load configuration from YAML """
    with open("../../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """ Seeding """
    seeding(config['seed'])

    """ Folders to save test images """
    result_folder_path = config['result_path_ensemble']
    create_dir(result_folder_path)
    images_result_path = os.path.join(result_folder_path, "images_result")
    create_dir(images_result_path)

    """ Load dataset """
    data_path = config['data_path']
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    """ Hyperparameters """
    H = config['image_height']
    W = config['image_width']

    """ Load model checkpoints """
    checkpoint_path_unet = config['checkpoint_path_unet']
    checkpoint_path_r2attunet = config['checkpoint_path_r2attunet']

    """ Check device """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    """ Import models """
    from unet_new import UNet
    from model.UNET.UNET_recurrent_model import R2AttUnet

    model_unet = UNet(dropout_rate=config['dropout_rate']).to(device)
    model_r2attunet = R2AttUnet(dropout_rate=config['dropout_rate']).to(device)

    """ Load model weights from checkpoints """
    model_unet.load_state_dict(torch.load(checkpoint_path_unet, map_location=device))
    model_r2attunet.load_state_dict(torch.load(checkpoint_path_r2attunet, map_location=device))
    model_unet.eval()
    model_r2attunet.eval()

    metrics_score = [0.0] * 5  # Initialize metrics

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Read and preprocess input image """
        original_image_path = x  # Store the original image path
        name = os.path.splitext(os.path.basename(x))[0]
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # (512, 512, 3)
        x = np.transpose(image, (2, 0, 1))  # (3, 512, 512)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)  # (1, 3, 512, 512)
        x = torch.from_numpy(x).float().to(device)

        """ Read and preprocess ground truth mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # (512, 512)
        y = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            """ Get predictions from both models """
            pred_unet = torch.sigmoid(model_unet(x))
            pred_r2attunet = torch.sigmoid(model_r2attunet(x))

            """ Apply OR operation to combine predictions """
            combined_pred = apply_or_operation(pred_unet, pred_r2attunet)

            """ Calculate metrics """
            score = calculate_metrics(y, combined_pred)
            metrics_score = list(map(add, metrics_score, score))

            """ Convert predictions to numpy for saving """
            pred_unet_np = (pred_unet.cpu().numpy()[0, 0, :, :] > 0.5) * 255
            pred_r2attunet_np = (pred_r2attunet.cpu().numpy()[0, 0, :, :] > 0.5) * 255
            combined_pred_np = (combined_pred.cpu().numpy()[0, 0, :, :] > 0.5) * 255

            # Convert boolean masks to uint8 for saving
            pred_unet_display = pred_unet_np.astype(np.uint8)
            pred_r2attunet_display = pred_r2attunet_np.astype(np.uint8)
            combined_display = combined_pred_np.astype(np.uint8)

            # Convert masks and predictions to 3 channels by repeating along the channel axis
            mask_display = mask[..., np.newaxis]  # (512, 512, 1)
            mask_display = np.repeat(mask_display, 3, axis=-1)  # Convert to (512, 512, 3)

            # Convert predictions to 3 channels by repeating along the channel axis
            pred_unet_display = np.repeat(pred_unet_display[..., np.newaxis], 3, axis=-1)  # (512, 512, 3)
            pred_r2attunet_display = np.repeat(pred_r2attunet_display[..., np.newaxis], 3, axis=-1)  # (512, 512, 3)
            combined_display = np.repeat(combined_display[..., np.newaxis], 3, axis=-1)  # (512, 512, 3)

            # Add labels to each display image
            image_labeled = add_label(image, "Original Image")
            mask_display_labeled = add_label(mask_display, "Mask")
            pred_unet_display_labeled = add_label(pred_unet_display, "UNet")
            pred_r2attunet_display_labeled = add_label(pred_r2attunet_display, "R2AttUNet")
            combined_display_labeled = add_label(combined_display, "OR Assemble")

            # Concatenate labeled images for comparison
            comparison_image = np.concatenate(
                (image_labeled, mask_display_labeled, pred_unet_display_labeled,
                 pred_r2attunet_display_labeled, combined_display_labeled), axis=1)

            # Save the concatenated comparison image with the same name as original image
            comparison_image_path = os.path.join(images_result_path, f"{name}.png")
            print(f"Saving image: {comparison_image_path}")
            cv2.imwrite(comparison_image_path, comparison_image)

    # Compute and print final average metrics
    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}"
    )
