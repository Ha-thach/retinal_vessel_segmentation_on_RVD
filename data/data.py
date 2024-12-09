import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class BuildDataset(Dataset):
    def __init__(self, images_path, masks_path, resize_to=(1024, 1024)):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.resize_to = resize_to

        # Define the resizing transform
        self.resize_transform = transforms.Resize(self.resize_to)

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image at index {index} could not be loaded. Check the path: {self.images_path[index]}")
        image = image / 255.0  # Normalize to [0, 1] range
        image = np.transpose(image, (2, 0, 1))  # Rearrange to (C, H, W)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        # Resize the image
        image = self.resize_transform(image)

        """ Reading mask """
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask at index {index} could not be loaded. Check the path: {self.masks_path[index]}")
        mask = mask / 255.0  # Normalize to [0, 1] range
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension (1, H, W)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        # Resize the mask
        mask = self.resize_transform(mask)

        return image, mask

    def __len__(self):
        return self.n_samples

