import os
import time
import yaml
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.data import BuildDataset
from utils.loss import DiceBCELoss
from utils.utils import seeding, create_dir, epoch_time, load_data

torch.cuda.empty_cache()

print(torch.cuda.memory_allocated())
print(torch.cuda.memory_reserved())

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(loader)
    return epoch_loss


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
    return epoch_loss


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

    """ Import the correct model """
    if model_name == "UNet":
        from model.UNET.unet import UNet

        model = UNet(dropout_rate=dropout_rate)  # Pass dropout rate to the model
    elif model_name == "R2AttUnet":
        from UNET_recurrent_model import R2AttUnet

        model = R2AttUnet(dropout_rate=dropout_rate)  # Ensure this model also takes dropout_rate
    else:
        print(f"Cannot import model")
        exit()

    """ Dataset and loader """
    train_dataset = BuildDataset(train_x, train_y)
    valid_dataset = BuildDataset(test_x, test_y)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    #print(train_dataset)
    #print(f'Train Loader {train_loader}')
    device = torch.device('cuda')
    model = model.to(device)
    #print(f'Model:{model}')
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    # loss_fn = DiceBCELoss()
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
        print(f'Epoch {epoch} :')
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Log metrics to TensorBoard """
        # Log both train and validation losses to TensorBoard with distinct tags
        writer.add_scalar('Train', train_loss, epoch + 1)
        writer.add_scalar('Validation', valid_loss, epoch + 1)

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
