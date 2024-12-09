import torch
from transformers import SamModel, SamConfig


def SAM_model_from_transformer(config):
    """Build and fine-tune SAM model for segmentation tasks."""
    # Load SAM model from registry
    model = SamModel.from_pretrained("facebook/sam-vit-base")

    # Freeze the encoder layers (vision encoder and prompt encoder)
    # for name, param in model.named_parameters():
    #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    #         param.requires_grad_(False)

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(model)

    # Set up optimizer to only fine-tune the mask decoder
    optimizer_class = getattr(torch.optim, config['optimizer'])
    optimizer = optimizer_class(
        model.mask_decoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler_class = getattr(torch.optim.lr_scheduler, config['learning_rate_scheduler'])
    if config['learning_rate_scheduler'] == 'ReduceLROnPlateau':
        scheduler = scheduler_class(optimizer, mode='min', patience=3, verbose=True)
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


    return model, optimizer, scheduler
