import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import yaml
from operator import add
from torch.utils.data import DataLoader
from model.SAM.sam_data import SAMDataset, initialize_processor
from model.SAM.SAM_mdel_using_transformers import SAM_model_from_transformer
#from transformers import SamProcessor
from utils.utils import create_dir, seeding, load_data
#from SAM.Customed_SAM_maskcoder import SAM_modified


def load_configuration():
    """Load configuration from YAML."""
    with open("../../config.yaml", "r") as file:
        return yaml.safe_load(file)


def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy() > 0.5
    y_pred = y_pred.cpu().numpy() > 0.5

    y_true = y_true.astype(np.uint8).reshape(-1)
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":
    # Load configuration and set random seed
    config = load_configuration()
    seeding(config['seed'])

    # Folders to save test images
    result_folder_path = config['result_path']
    images_result_path = os.path.join(result_folder_path, "images_result")
    mask_result_path = os.path.join(result_folder_path, "predicted_mask_aug_no_equal_no_nor_result")
    create_dir(images_result_path)
    create_dir(mask_result_path)
    results_csv_path = os.path.join(result_folder_path, "results.csv")

    # Load dataset
    data_path = config['data_path']
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    # Hyperparameters
    H = config['image_height']
    W = config['image_width']
    batch_size = config['batch_size']
    size = (W, H)
    model_name = config['model']
    dropout_rate = config['dropout_rate']
    checkpoint_path = os.path.join(result_folder_path, "best_model.pth")

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize processor
    processor = initialize_processor()

    # Prepare dataset and DataLoader
    test_dataset = SAMDataset(test_x, test_y, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load Model
    model, optimizer, scheduler = SAM_model_from_transformer(config)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    # Iterate over test set in batches
    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, test_batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            try:
                # Forward pass
                outputs = model(
                    pixel_values=test_batch["pixel_values"].to(device),
                    input_boxes=test_batch["input_boxes"].to(device),
                    multimask_output=False
                )

                # Extract predicted masks
                predicted_test_masks = outputs.pred_masks.squeeze(1)  # Shape: (batch_size, 256, 256)
                ground_truth_masks = test_batch["mask"].float().to(device)  # Shape: (batch_size, 1, 256, 256)

                for i in range(len(test_batch["mask"])):  # Iterate through the batch
                    # Calculate metrics for each sample
                    score = calculate_metrics(ground_truth_masks[i], predicted_test_masks[i])
                    metrics_score = list(map(add, metrics_score, score))

                    #Just-single predicted mask
                    pred_mask_prob = torch.sigmoid(predicted_test_masks[i])  # Shape: (1, 256, 256)
                    pred_mask = (pred_mask_prob.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
                    pred_mask_rescaled = cv2.resize(pred_mask * 255, (512, 512), interpolation=cv2.INTER_NEAREST)
                    output_image_path = os.path.join(mask_result_path,f"{batch_idx * batch_size + i}.png")
                    cv2.imwrite(output_image_path, pred_mask_rescaled)

                    #Concatenate images
                    # Apply sigmoid to get probabilities
                    # Apply sigmoid to get probabilities
                    pred_mask_prob = torch.sigmoid(predicted_test_masks[i])  # Shape: (1, 256, 256)
                    #print(f"pred_mask_prob shape: {pred_mask_prob.shape}")  # (1, 256, 256)

                    # Remove the batch dimension (shape becomes (256, 256))
                    pred_mask = (pred_mask_prob.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
                    #print(f"pred_mask shape: {pred_mask.shape}")  # (256, 256)

                    # Resize pred_mask and ensure it has 3 channels
                    pred_mask_rescaled = cv2.resize(pred_mask * 255, (1024, 1024),
                                                    interpolation=cv2.INTER_NEAREST)  # Scale to 255 for visual display
                    pred_mask_rescaled = np.stack([pred_mask_rescaled] * 3, axis=-1)  # Shape: (1024, 1024, 3)
                    #print(f"pred_mask_rescaled shape: {pred_mask_rescaled.shape}")  # Shape: (1024, 1024, 3)

                    # Ensure ground truth mask is in the right format
                    ori_mask = ground_truth_masks[i].cpu().numpy().squeeze()  # Shape: (256, 256)
                    ori_mask_rescaled = cv2.resize(ori_mask * 255, (1024, 1024))  # Ensure it's [0, 255]
                    ori_mask_rescaled = np.stack([ori_mask_rescaled] * 3, axis=-1).astype(np.uint8)
                    #print(f"ori_mask shape: {ori_mask.shape}")  # Print shape

                    # Resize original image and masks
                    original_image = test_batch["pixel_values"][i].cpu().numpy().transpose(1, 2, 0)  # Shape: (256, 256, 3)
                    original_image = (original_image * 255).astype(np.uint8)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
                    # Prepare a separator line
                    line = np.ones((1024, 10, 3), dtype=np.uint8) * 128  # Separator line
                    #print(f"line shape: {line.shape}")  # Print shape

                    # Concatenate images and masks
                    cat_images = np.concatenate(
                        [original_image, line, ori_mask_rescaled, line, pred_mask_rescaled],
                        axis=1
                    )
                    #print(f"cat_images shape: {cat_images.shape}")  # Print shape

                    # Save the concatenated image
                    output_image_path = os.path.join(images_result_path,
                                                     f"concatenated_{batch_idx * batch_size + i}.png")
                    cv2.imwrite(output_image_path, cat_images)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    # Calculate average metrics
    num_samples = len(test_x)
    jaccard = metrics_score[0] / num_samples
    f1 = metrics_score[1] / num_samples
    recall = metrics_score[2] / num_samples
    precision = metrics_score[3] / num_samples
    acc = metrics_score[4] / num_samples

    print(f"Jaccard: {jaccard:.4f} - F1: {f1:.4f} - Recall: {recall:.4f} - Precision: {precision:.4f} - Acc: {acc:.4f}")

    # Calculate FPS if time measurements are available
    fps = 1 / np.mean(time_taken) if time_taken else 0
    print("FPS: ", fps)

    # Save the evaluation metrics to a CSV file with a new section
    with open(results_csv_path, 'a') as f:
        f.write("\n\n### New Evaluation Section ###\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Jaccard: {jaccard:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")