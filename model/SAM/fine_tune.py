import os
import time
import yaml
import csv
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.SAM.sam_data import SAMDataset, initialize_processor
from transformers import SamModel
from utils.loss import DiceBCELoss
from utils.utils import seeding, create_dir, epoch_time, load_data
from tqdm import tqdm
torch.cuda.empty_cache()

print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

def train(model, loader, optimizer, loss_fn, device, gradient_scaling_factor=1.0):
    """Train the model for one epoch with scaled gradients."""
    losses = []  # List to store batch losses
    model.train()  # Set the model to training mode

    for i, batch in enumerate(tqdm(loader)):
        # Forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device))
        #print(outputs)

        # Compute loss
        #predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["mask"].float().to(device).unsqueeze(1)
        ground_truth_masks_resized = torch.nn.functional.interpolate(
            ground_truth_masks,
            size=(outputs.size(2), outputs.size(3)),  # Use output's height and width
            mode='bilinear',
            align_corners=False
        )
        # print("Output shape:", outputs.shape)
        # print("Ground truth shape:", ground_truth_masks.shape)

        loss = loss_fn(outputs, ground_truth_masks_resized)

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
            outputs = model(pixel_values=val_batch["pixel_values"].to(device))

            # Calculate validation loss
            ground_truth_masks = val_batch["mask"].float().to(device).unsqueeze(1)
            ground_truth_masks_resized = torch.nn.functional.interpolate(
                ground_truth_masks,
                size=(outputs.size(2), outputs.size(3)),  # Use output's height and width
                mode='bilinear',
                align_corners=False
            )

            val_loss = loss_fn(outputs, ground_truth_masks_resized)

            losses.append(val_loss.item())  # Store validation loss for this batch

    # Calculate average loss
    epoch_loss = sum(losses) / len(losses) if losses else 0.0  # Handle empty loader
    return epoch_loss
def check_trainable_status(model):
    """Check and print whether each layer in the model is trainable or not."""
    for name, module in model.named_modules():
        if hasattr(module, 'parameters'):
            # Check if any parameter in the module is trainable
            is_trainable = any(param.requires_grad for param in module.parameters())
            # Print the status
            #print(f"{name}: {'Trainable' if is_trainable else 'Not trainable'}")

def freeze_layers(model, freeze_until_layer=6, unfreeze_from_layer=6, neck=True):
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

import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_rate=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Apply dropout
        x = self.dropout(x)

        return x

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, dropout_rate=0.0):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c, out_c, dropout_rate)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.conv(x)
        return x
class SAM_UNet(nn.Module):
    def __init__(self, num_classes):
        super(SAM_UNet, self).__init__()

        # Load the pre-trained SAM model
        self.vision_encoder = SamModel.from_pretrained("facebook/sam-vit-base").vision_encoder

        # U-Net decoder layers
        self.d1=decoder_block(256, 128)
        self.d2=decoder_block(128,64)
        self.d3=decoder_block(64,32)
        self.d4=decoder_block(32, 16)


        self.output = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, pixel_values):
        # Pass input through the encoder
        encoder_output = self.vision_encoder(pixel_values)

        # Print the encoder output to debug its structure
        #print("Encoder output:", encoder_output)

        # Access the appropriate attribute from the encoder output
        if hasattr(encoder_output, 'last_hidden_state'):
            x = encoder_output.last_hidden_state  # Use last_hidden_state or whatever output is correct
        else:
            raise ValueError("Expected encoder_output to have 'last_hidden_state' attribute, but it does not.")

        # Check if the extracted tensor is None
        if x is None:
            raise ValueError("Extracted tensor from encoder_output is None.")

        # Pass the extracted tensor through the upsampling layers
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)


        x = self.output(x)

        return x



if __name__ == "__main__":
    """ Load configuration from YAML """
    with open("../../config.yaml", "r") as file:
        config = yaml.safe_load(file)

    """ Seeding """
    seeding(config['seed'])

    """ Load dataset """
    data_path = config['data_path']  # Assuming the path is provided in config
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(test_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = config['image_height']
    W = config['image_width']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    print(f"Numbers of epochs: {num_epochs}")
    lr = config['learning_rate']
    print(f"lr: {lr}")
    result_folder_path = config['result_path']
    create_dir(result_folder_path)

    dropout_rate = config['dropout_rate']  # Read dropout rate from YAML
    print(f"Drop out: {dropout_rate}")
    model_name = config['model']

    # Define paths for the checkpoint and CSV file
    checkpoint_path = os.path.join(result_folder_path, "checkpoint.pth")
    results_csv_path = os.path.join(result_folder_path, "results.csv")

    processor = initialize_processor()
    train_dataset = SAMDataset(train_x, train_y, processor=processor)
    valid_dataset = SAMDataset(test_x, test_y, processor=processor)
    # train_dataset = BuildDataset(train_x, train_y)
    # valid_dataset = BuildDataset(test_x, test_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    sample =train_dataset[0]
    #print(sample)

    # Instantiate the model with the desired number of output classes
    num_classes = 1  # Change according to your task
    model = SAM_UNet(num_classes)
    #print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    freeze_layers(model, freeze_until_layer=6, unfreeze_from_layer=6, neck=True)
    print(model)
    # Check the trainable status of layers
    #print("\nTrainable Status of Layers:")
    check_trainable_status(model)

    # Move the model to GPU if available


    # Set up optimizer to only fine-tune the mask decoder
    optimizer_class = getattr(torch.optim, config['optimizer'])
    optimizer = optimizer_class(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler_class = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler'])
    if config['learning_rate_scheduler'] == 'ReduceLROnPlateau':
        scheduler = scheduler_class(optimizer, mode='min', patience=config['patience'], verbose=True)
    elif config['learning_rate_scheduler'] == 'CosineAnnealingLR':
        scheduler = scheduler_class(optimizer, T_max=config['T_max'], eta_min=config.get('eta_min', 0))
    elif config['learning_rate_scheduler'] == 'OneCycleLR':
        scheduler = scheduler_class(
            optimizer,
            max_lr=config['max_lr'],
            steps_per_epoch=config['steps_per_epoch'],
            epochs=config['num_epochs']
        )
    else:
        scheduler = scheduler_class(optimizer, **config['scheduler_params'])
    loss_fn = DiceBCELoss()

    """ Initialize SummaryWriter """
    writer = SummaryWriter(log_dir=result_folder_path)

    """ Initialize CSV file """
    with open(results_csv_path, 'w', newline='') as csvfile:
        writer_csv = csv.writer(csvfile)
        writer_csv.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Model Saved'])

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Log metrics to TensorBoard """
        writer.add_scalar('Train Loss', train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', valid_loss, epoch + 1)

        model_saved = False

        """ Saving the model """
        if valid_loss < best_valid_loss:
            print(
                f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

            model_saved = True

        """ Save results to CSV """
        with open(results_csv_path, 'a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([epoch + 1, train_loss, valid_loss, model_saved])

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
        writer.flush()

    writer.close()
