import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import SamProcessor


class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.transforms = self.transforms_f  # Set the Transform function for processing

    def transforms_f(self, image_path: str, mask_path: str):
        # Define transformations for both image and mask
        img_transforms = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Resize image to 1024x1024
            transforms.ToTensor()  # Convert to tensor (float, [0, 1])
        ])

        mask_transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize mask to 1024x1024
            transforms.ToTensor()  # Convert mask to tensor (float, [0, 1])
        ])

        # Load the image and apply transformations
        image = Image.open(image_path).convert("RGB")  # Load color image
        transformed_img = img_transforms(image)

        # Load the mask and apply transformations
        mask = Image.open(mask_path).convert("L")  # Load grayscale mask
        transformed_mask = mask_transforms(mask)

        # Return the processed image and mask in a dictionary
        return {'img': transformed_img, 'label': transformed_mask}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Apply transformations to image and mask
        data_dict = self.transforms(image_path, mask_path)  # Dictionary of tensors {'img': ..., 'label': ...}
        image = data_dict['img'].squeeze()  # Keep the image as float32, shape: (C, H, W)
        mask = data_dict['label'].squeeze()  # Keep the mask as float32, shape: (H, W)

        # Print image shape for debugging
        #print(f'check image in float32 {image.shape}')  # Debugging print

        # Function to get bounding box prompt from the mask
        prompt = self.get_bounding_box(mask.numpy())  # Convert mask to numpy for bbox calculation

        # Prepare inputs for SAM processor (SAM works on float32, no need to convert to uint8)
        inputs = self.processor(image.permute(1, 2, 0).numpy(), input_boxes=[[prompt]], return_tensors="pt")

        # Remove the batch dimension added by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add the ground truth mask to inputs
        inputs["mask"] = mask  # Keep mask as float32
        # Add filename to the output
        # Get the filename from the image path for tracking purposes
        filename = os.path.basename(image_path)
        inputs["filename"] = filename  # Include filename in the output

        return inputs

    def get_bounding_box(self, mask):
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

def initialize_processor():
    """Initialize and modify the SAM processor."""
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    # Modify processor attributes as needed
    processor.image_processor.do_convert_rgb = False
    processor.image_processor.do_normalize = False
    processor.image_processor.do_pad = False
    processor.image_processor.do_rescale = False
    processor.image_processor.do_resize = True
    processor.image_processor.image_mean = [123.675, 116.28, 103.53]  # Custom mean values
    processor.image_processor.image_std = [58.395, 57.12, 57.375]  # Custom std values
    processor.image_processor.mask_pad_size = {"height": 0, "width": 0}  # Custom mask padding size
    processor.image_processor.mask_size = {"longest_edge": 256}  # Custom mask size
    processor.image_processor.pad_size = {"height": 1024 , "width": 1024}  # Custom padding size
    processor.image_processor.resample = 2  # Custom resample value
    processor.image_processor.rescale_factor = 0.00392156862745098  # Custom rescale factor
    processor.image_processor.size = {"longest_edge": 1024}  # Custom image resizing based on longest edge

    # Add other modifications as needed
    return processor