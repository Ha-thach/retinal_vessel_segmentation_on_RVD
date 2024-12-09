import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.utils import seeding, create_dir, load_data
from model.SAM.sam_data import SAMDataset
from build_fine_tune_model import build_fine_tune_model
from transformers import SamProcessor

# Load the pre-trained SAM processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Modify attributes of the image processor
processor.image_processor.do_convert_rgb = False
processor.image_processor.do_normalize = False
processor.image_processor.do_pad = False
processor.image_processor.do_rescale = False
processor.image_processor.do_resize = True
processor.image_processor.image_mean = [123.675, 116.28, 103.53]  # Custom mean values
processor.image_processor.image_std = [58.395, 57.12, 57.375]    # Custom std values
processor.image_processor.mask_pad_size = {"height": 0, "width": 0}  # Custom mask padding size
processor.image_processor.mask_size = {"longest_edge": 1024}  # Custom mask size
processor.image_processor.pad_size = {"height": 1024, "width": 1024}  # Custom padding size
processor.image_processor.resample = 2  # Custom resample value
processor.image_processor.rescale_factor = 0.00392156862745098  # Custom rescale factor
processor.image_processor.size = {"longest_edge": 1024}  # Custom image resizing based on longest edge
print(processor)

# Save modified processor configuration (if needed)
processor.save_pretrained("/data/nhthach/project/results/binary_full/model1-SAM-10ep-dr.0_lre-4/processor")
print(processor)

torch.cuda.empty_cache()


def get_bounding_box(mask):
    """
    Get bounding box (xmin, ymin, xmax, ymax) for the given mask.
    """
    # Find non-zero values in the mask
    coords = np.column_stack(np.where(mask > 0))

    if len(coords) == 0:
        return [0, 0, 0, 0]  # Empty mask case

    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)

    return [xmin, ymin, xmax, ymax]
def train(model, loader, optimizer, loss_fn, device):
    """Train the model for one epoch."""
    epoch_loss = 0.0
    model.train()

    for data in loader:
        # Ensure each image is correctly formatted as 4D: [batch_size, channels, height, width]
        # Assuming `data[0]['image']` is [batch_size, 3, height, width]
        x = data[0]['image'].to(device)  # Move image tensor to device
        batched_input = [{'image': x}]  # Wrap image tensor in a list of dictionaries
        y = data[0]['mask'].to(device)
        optimizer.zero_grad()
        y_pred = model(batched_input, multimask_output=False)  # Model expects list of dicts
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(loader)

    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    """Evaluate the model on the validation set."""
    epoch_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0]['image'].to(device)  # Ensure batch input is correctly formatted
            batched_input = [{'image': x}]  # Wrap image tensor as a dict in a list
            print(f'Image:{batched_input}')
            y = data[0]['mask'].to(device)
            y_pred = model(batched_input[1:], multimask_output=False)

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
    result_folder_path = config['result_path']
    dropout_rate = config['dropout_rate']  # Read dropout rate from YAML

    create_dir(result_folder_path)

    # Define paths for the checkpoint and CSV file
    checkpoint_path = os.path.join(result_folder_path, "checkpoint.pth")
    results_csv_path = os.path.join(result_folder_path, "results.csv")

    """ Load and fine-tune SAM model """
    model, optimizer, scheduler = build_fine_tune_model(config)

    """ Dataset and DataLoader """
    train_dataset = SAMDataset(train_x, train_y, processor=processor)

    valid_dataset = SAMDataset(test_x, test_y, processor=processor)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = batch_size, shuffle = False)
    print(train_dataset)
    print(train_loader.__dict__)
    # model = SamModel.from_pretrained("facebook/sam-vit-base")
    #
    # # make sure we only compute gradients for mask decoder (encoder weights are frozen)
    # for name, param in model.named_parameters():
    #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #         print(name)
    #         param.requires_grad_(False)
    example = train_dataset[0]
    print(f'Example{example}')
    # for k, v in example.items():
    #     print(f'K,v shape{k, v.shape}')
    # xmin, ymin, xmax, ymax = get_bounding_box(example['mask'])
    #
    # fig, axs = plt.subplots(1, 2)
    #
    # axs[0].imshow(example['pixel_values'].permute(1, 2, 0))  # Color image
    # axs[0].axis('off')
    #
    # axs[1].imshow(example['mask'], cmap='gray')  # Grayscale image
    # axs[1].axis('off')
    #
    # plt.tight_layout()
    # plt.show()

