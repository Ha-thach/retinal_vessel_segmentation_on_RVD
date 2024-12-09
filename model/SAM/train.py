

import os
import time
import yaml
import torch
import csv
from torch.utils.data import DataLoader
from utils.utils import seeding, create_dir, load_data
from model.SAM.sam_data import SAMDataset, initialize_processor
#from SAM.SAM_mdel_using_transformers import SAM_model_from_transformer
from Customed_SAM_maskcoder import SAM_modified
from utils.loss import DiceBCELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#from SAM.SAM_w_UNET_decoder import SAM_modified
def check_trainable_status(model):
    """Check and print whether each layer in the model is trainable or not."""
    for name, module in model.named_modules():
        if hasattr(module, 'parameters'):
            # Check if any parameter in the module is trainable
            is_trainable = any(param.requires_grad for param in module.parameters())
            # Print the status
            print(f"{name}: {'Trainable' if is_trainable else 'Not trainable'}")

def freeze_layers(model, freeze_until_layer=8, unfreeze_from_layer=8, neck=True):
    """Freeze and unfreeze layers in the vision encoder."""
    # Freeze layers in the vision encoder
    for i, layer in enumerate(model.vision_encoder.layers):
        if i < freeze_until_layer:
            for param in layer.parameters():
                param.requires_grad = False
        elif i >= unfreeze_from_layer:
            for param in layer.parameters():
                param.requires_grad = True

    # Optionally unfreeze the neck layer
    if neck:
        for param in model.vision_encoder.neck.parameters():
            param.requires_grad = True


def train(model, loader, optimizer, loss_fn, device, gradient_scaling_factor=1.0):
    """Train the model for one epoch with scaled gradients."""
    losses = []  # List to store batch losses
    model.train()  # Set the model to training mode

    for i, batch in enumerate(tqdm(loader)):
        # Forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # Compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["mask"].float().to(device)

        loss = loss_fn(predicted_masks, ground_truth_masks.unsqueeze(1))

        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()  # Backpropagation

        # Scale gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data *= gradient_scaling_factor

        optimizer.step()  # Update parameters

        # Store loss for this batch
        losses.append(loss.item())

    # Calculate average loss for the epoch
    epoch_loss = sum(losses) / len(losses) if losses else 0.0
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    """Evaluate the model on the validation set."""
    losses = []  # List to store validation losses
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for val_batch in tqdm(loader):  # Iterate over validation batches
            # Forward pass
            outputs = model(pixel_values=val_batch["pixel_values"].to(device),
                            input_boxes=val_batch["input_boxes"].to(device),
                            multimask_output=False)

            # Calculate validation loss
            predicted_val_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = val_batch["mask"].float().to(device)
            val_loss = loss_fn(predicted_val_masks, ground_truth_masks.unsqueeze(1))

            losses.append(val_loss.item())  # Store validation loss for this batch

    # Calculate average loss
    epoch_loss = sum(losses) / len(losses) if losses else 0.0  # Handle empty loader
    return epoch_loss

def load_configuration():
    """Load configuration from YAML."""
    with open("../../config.yaml", "r") as file:
        return yaml.safe_load(file)

def get_device():
    """Return the appropriate device (GPU/CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_results(writer, epoch, train_loss, valid_loss):
    """Log results to TensorBoard and CSV."""
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', valid_loss, epoch)

def main():
    """Main function to execute the training and evaluation loop."""
    # Load configuration and set seed
    config = load_configuration()
    seeding(config['seed'])

    # Load dataset
    data_path = config['data_path']
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    # Print dataset sizes
    print(f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(test_x)}")

    # Hyperparameters
    batch_size = config['batch_size']
    print(batch_size)
    num_epochs = config['num_epochs']
    result_folder_path = config['result_path']
    create_dir(result_folder_path)

    # Load model and optimizer
    model, optimizer, scheduler = SAM_modified(config)

    # Freeze and unfreeze layers in the model
    freeze_layers(model, freeze_until_layer=8, unfreeze_from_layer=8, neck=True)

    # Check the trainable status of layers
    print("\nTrainable Status of Layers:")
    check_trainable_status(model)
    # Initialize data loaders
    processor = initialize_processor()
    train_dataset = SAMDataset(train_x, train_y, processor=processor)
    valid_dataset = SAMDataset(test_x, test_y, processor=processor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    # Move model to device
    device = get_device()
    model.to(device)


    # Define loss function and initialize TensorBoard writer
    loss_fn = DiceBCELoss()
    writer = SummaryWriter(log_dir=result_folder_path)

    # Initialize CSV file for logging results
    results_csv_path = os.path.join(result_folder_path, "results.csv")
    with open(results_csv_path, 'w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Model Saved'])

    best_valid_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()

        # Train and evaluate
        train_loss = train(model, train_loader, optimizer, loss_fn, device, gradient_scaling_factor=0.1)  # Example scaling factor
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        # Log results
        log_results(writer, epoch, train_loss, valid_loss)
        model_saved = False

        # Save model checkpoint if validation loss improved
        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}.")
            best_valid_loss = valid_loss
            model_saved = True
            torch.save(model.state_dict(), os.path.join(result_folder_path, "best_model.pth"))

        # Log to CSV
        with open(results_csv_path, 'a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([epoch + 1, train_loss, valid_loss, model_saved])

        # Print epoch results
        epoch_duration = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {valid_loss:.4f} | Time: {epoch_duration:.2f}s")

    writer.close()  # Close the TensorBoard writer at the end

if __name__ == "__main__":
    main()

