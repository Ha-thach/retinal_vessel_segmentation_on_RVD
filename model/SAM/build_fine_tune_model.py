import torch
from segment_anything import sam_model_registry

def build_fine_tune_model(config):
    """Build and fine-tune SAM model for segmentation tasks."""
    # Load SAM model from registry
    sam_type = config['sam_type']
    sam_checkpoint_path = config['sam_checkpoint_path']
    model = sam_model_registry[sam_type](checkpoint=sam_checkpoint_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Freeze the image encoder parameters
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Set up optimizer to only fine-tune the mask decoder
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=config['learning_rate'])

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    return model, optimizer, scheduler
