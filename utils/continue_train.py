import os
import time
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.SAM.sam_data import SAMDataset, initialize_processor
from loss import DiceBCELoss
from utils import seeding, epoch_time, load_data
from model.SAM.fine_tune import train, evaluate
import csv
from model.SAM.fine_tune import SAM_UNet
def continue_training(model, optimizer, scheduler, train_loader, valid_loader, loss_fn, config, start_epoch=0):
    """Continue training the model from a given epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up paths
    checkpoint_path = "/data/nhthach/project/results/binary_0.15/SAM-Unetdecoder_aug_no_equal_no_norm_lre-4_Adamw.dr_0.3_ep30_Cosine_train_feature_extraction_free0-6_4decoder/checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint.keys())
    results_csv_path = os.path.join(config['result_path'], "results.csv")
    num_epochs = config['num_epochs']
    num_epoch_more = config['num_epoch_more']
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config['result_path'])

    # Track the best validation loss
    best_valid_loss = float("inf")
    model_saved=False
    # Load previous results if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        model_saved=True
    # Continue training
    for epoch in range(num_epochs, (num_epochs+ num_epoch_more)):
        start_time = time.time()

        # Training and validation
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        # Log metrics to TensorBoard
        writer.add_scalar('Train Loss', train_loss, epoch + 1)
        writer.add_scalar('Validation Loss', valid_loss, epoch + 1)

        # Save the model if validation loss improves
        if valid_loss < best_valid_loss:
            print(f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint.")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        # Save results to CSV
        with open(results_csv_path, 'a', newline='') as csvfile:
            writer_csv = csv.writer(csvfile)
            writer_csv.writerow([epoch + 1, train_loss, valid_loss, model_saved])

        # Update the scheduler
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_loss)
        else:
            scheduler.step()

        # Print epoch time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

        writer.flush()

    writer.close()


# Load your config file
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set up seeding, dataset, model, optimizer, etc.
seeding(config['seed'])
(train_x, train_y), (test_x, test_y) = load_data(config['data_path'])
processor = initialize_processor()
train_dataset = SAMDataset(train_x, train_y, processor=processor)
valid_dataset = SAMDataset(test_x, test_y, processor=processor)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

# Initialize model, optimizer, scheduler, and loss function
num_classes = 1
model = SAM_UNet(num_classes)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
loss_fn = DiceBCELoss()

# Continue training from where you left off
continue_training(model, optimizer, scheduler, train_loader, valid_loader, loss_fn, config)
