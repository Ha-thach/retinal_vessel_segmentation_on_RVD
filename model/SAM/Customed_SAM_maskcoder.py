
from transformers.models.sam.modeling_sam import SamFeedForward
from transformers import SamMaskDecoderConfig, SamModel


import torch
import torch.nn as nn
from typing import Tuple

class CustomMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = self.num_multimask_outputs + 1

        # Embeddings for IoU prediction and mask tokens
        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        # U-Net-like layers for upsampling
        self.upconv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 2, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(self.hidden_size // 2, self.hidden_size // 4, kernel_size=2, stride=2)

        # Convolution blocks to replace attention layers
        self.conv_block1 = self._conv_block(self.hidden_size // 2, self.hidden_size // 2)
        self.conv_block2 = self._conv_block(self.hidden_size // 4, self.hidden_size // 4)

        # Prediction head for IoU
        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

        # Additional layer for generating the attention map (you may adjust the parameters)
        self.attention_conv = nn.Conv2d(self.hidden_size // 4, 1, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int):
        """Create a convolutional block with BatchNorm and ReLU activation."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        image_positional_embeddings: torch.Tensor = None,
        sparse_prompt_embeddings: torch.Tensor = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
        output_attentions: bool = None,
        skip_connections: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`): Embeddings from the image encoder.
            dense_prompt_embeddings (`torch.Tensor`): Embeddings of the mask inputs.
            multimask_output (bool): Whether to return multiple masks or a single mask.
            image_positional_embeddings (`torch.Tensor`, optional): Optional positional embeddings.
            sparse_prompt_embeddings (`torch.Tensor`, optional): Embeddings of the points and boxes.
            skip_connections (Tuple[torch.Tensor, torch.Tensor], optional): Skip connections from the encoder.
        """
        batch_size, _, height, width = image_embeddings.shape

        # Concatenate embeddings
        image_embeddings = image_embeddings + dense_prompt_embeddings

        # U-Net-like upsampling
        upsampled1 = self.upconv1(image_embeddings)
        if skip_connections is not None:
            upsampled1 = torch.cat((upsampled1, skip_connections[0]), dim=1)  # Use first skip connection
        upsampled1 = self.conv_block1(upsampled1)

        # Ensure upsampled1 is valid before proceeding
        if upsampled1 is not None:
            upsampled2 = self.upconv2(upsampled1)
            if skip_connections is not None:
                upsampled2 = torch.cat((upsampled2, skip_connections[1]), dim=1)  # Use second skip connection
            masks = self.conv_block2(upsampled2)
        else:
            raise ValueError("upsampled1 is None, check your U-Net upsampling process.")

        # Generate mask quality predictions
        iou_token_out = self.iou_token.weight
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Generate attention map from the last upsampled layer
        attention_map = self.attention_conv(upsampled2)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)  # Use all masks except the first one
        else:
            mask_slice = slice(0, 1)  # Use only the first mask

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred, attention_map  # Return all three outputs


def SAM_modified(config):
    """Build and fine-tune SAM model for segmentation tasks."""
    # Load SAM model from registry
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # Freeze the encoder layers
    # for name, param in model.named_parameters():
    #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #         param.requires_grad_(False)

    # Replace the mask decoder with the modified U-Net decoder
    decoder_config = SamMaskDecoderConfig(
        hidden_size=256,
        hidden_act='relu',
        mlp_dim=2048,
        iou_head_depth=3,
        iou_head_hidden_dim=256
    )
    print(f'Decoder config: {decoder_config}')
    model.mask_decoder = CustomMaskDecoder(decoder_config)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(
        model.mask_decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, verbose=True
    )

    return model, optimizer, scheduler
