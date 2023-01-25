import torch
import lightning
from pytorch_lightning.loggers import WandbLogger

from .train_module import TrainModule
from data.imagenet import ImageNetModule

def train(
    model:torch.nn.Module,
    config:object):
    """ Train a model on ImageNet using PyTorch Lightning. Uses the state-of-the-art data augmentation strategies.
    
    Args:
        model: (torch.nn.Module) torch model to train
        root: (string) path to the ImageNet root directory
        save_dir: (string) path to save model checkpoints
    """
    # Initialize data and model
    dataloader = ImageNetModule(config.root, resize_size=config.resize_size, crop_size=config.crop_size, batch_size=config.batch_size, workers=config.workers)
    train_module = TrainModule(model, optimizer=config.optimizer, learning_rate=config.learning_rate, scheduler=config.scheduler, num_gpus=config.num_gpus)

    
    if config.use_wandb:
        logger = WandbLogger(
            project=config.project_name,
            save_dir=config.save_dir
        )
    else: logger = True # Use TensorBoard Logger

    if config.num_gpus > 1: strategy = 'ddp'
    else: strategy = None

    trainer = lightning.Trainer(
        accelerator=config.device, 
        devices=config.num_gpus, 
        max_epochs=config.epochs,
        default_root_dir=config.save_dir,
        logger=logger,
        strategy=strategy,
        enable_progress_bar= not config.use_wandb,
        log_every_n_steps=50,
    )

    # Train the model
    trainer.fit(train_module, dataloader)

def validate(model, ckpt_path, device='gpu', num_gpus:int=1):
    dataloader = ImageNetModule("/home/hyperai1/jhsong/Data/ImageNet", resize_size=256, crop_size=224)
    train_module = TrainModule(model).load_from_checkpoint(ckpt_path)

    trainer = lightning.Trainer(
        accelerator=device, 
        devices=num_gpus, 
        enable_progress_bar= True
    )

    trainer.validate(train_module, dataloader)
    return