
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SamMaskDecoderConfig, SamModel
import yaml


class ModifiedUNetDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig, out_channels=1):
        super(ModifiedUNetDecoder, self).__init__()

        self.hidden_size = config.hidden_size  # Typically 256
        self.mlp_dim = config.mlp_dim  # Typically 2048
        self.hidden_act = config.hidden_act  # Activation function, e.g., "relu"
        self.iou_head_depth = config.iou_head_depth
        self.iou_head_hidden_dim = config.iou_head_hidden_dim

        # Define up-sampling layers
        self.up1 = nn.ConvTranspose2d(self.hidden_size, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        # Define convolutional layers
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, out_channels, kernel_size=1)

        # Define the activation function based on the configuration
        self.activation = self.get_activation_function(config.hidden_act)

        # Optional IoU head for output refinement
        self.iou_head = self._build_iou_head(config.iou_head_depth, config.iou_head_hidden_dim)

    def get_activation_function(self, activation_name):
        """Return the activation function based on the provided name."""
        if activation_name == "relu":
            return F.relu
        elif activation_name == "gelu":
            return F.gelu
        elif activation_name == "sigmoid":
            return torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

    def _build_iou_head(self, depth, hidden_dim):
        """Build the IoU head module with given depth and hidden dimension."""
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())  # Using ReLU as a common choice for hidden layers
        layers.append(nn.Linear(hidden_dim, 1))  # Output layer for IoU prediction
        return nn.Sequential(*layers)

    def forward(self, image_embeddings, *args, **kwargs):
        """Forward pass through the modified U-Net decoder."""
        # Assuming `image_embeddings` has shape [batch_size, hidden_size, H, W]
        x = self.up1(image_embeddings)
        x = self.activation(self.conv1(x))
        x = self.up2(x)
        x = self.activation(self.conv2(x))
        x = self.up3(x)
        x = self.activation(self.conv3(x))
        x = self.up4(x)
        x = self.conv4(x)

        # IoU prediction (if using IoU head)
        iou_prediction = self.iou_head(x.flatten(1)) if self.iou_head else None

        return x, iou_prediction


def SAM_modified(config):
    """Build and fine-tune SAM model for segmentation tasks."""
    # Load SAM model from registry
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # Freeze the encoder layers
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Replace the mask decoder with the modified U-Net decoder
    decoder_config = SamMaskDecoderConfig(hidden_size=256, hidden_act='relu', mlp_dim=2048,
                                          iou_head_depth=3, iou_head_hidden_dim=256)
    print(f'Decoder config:{decoder_config}')
    model.mask_decoder = ModifiedUNetDecoder(decoder_config, out_channels=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)


    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
#
# # Assuming train_loader is defined and provides (images, masks)
# #train_model(model, optimizer, scheduler, train_loader, config['num_epochs'], device)

    return model, optimizer, scheduler

def load_configuration():
    """Load configuration from YAML."""
    with open("../../config.yaml", "r") as file:
        return yaml.safe_load(file)


def get_device():
    """Return the appropriate device (GPU/CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = load_configuration()
    # Hyperparameters
batch_size = config['batch_size']
print(batch_size)
num_epochs = config['num_epochs']



    # Load model and optimizer
model, optimizer, scheduler = SAM_modified(config)
print(model)








