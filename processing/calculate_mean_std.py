import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define a custom dataset
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Define the directory containing the images
data_dir = '/data/nhthach/project/DATA/RETINAL/Full/IMAGE'

# Define the transformation to convert images to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load the dataset
dataset = ImageDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

# Function to calculate mean and std
"""calculates the mean and standard deviation for each color channel (Red, Green, Blue) of the color images in the dataset
"""
def calculate_mean_std(dataloader):
    mean = torch.zeros(3)  # Assuming 3 channels (RGB)
    std = torch.zeros(3)
    total_images = 0

    for images in dataloader:
        batch_images = images.size(0)  # Number of images in the batch
        total_images += batch_images

        # Mean and std for each channel
        mean += images.mean([0, 2, 3]) * batch_images
        std += images.std([0, 2, 3]) * batch_images

    mean /= total_images
    std /= total_images

    return mean, std

# Calculate mean and std
mean, std = calculate_mean_std(dataloader)
print(f'Mean: {mean}')
print(f'Std: {std}')

#Mean: tensor([0.3807, 0.1883, 0.0998])
#Std: tensor([0.2927, 0.1607, 0.1053])