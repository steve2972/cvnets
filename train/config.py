class Config(object):
    # Set the root path to the ImageNet Dataset
    root:str = "/home/hyperai1/jhsong/Data/ImageNet"

    # Training parameters.
    epochs:int = 150
    batch_size:int = 256

    # Number of workers when loading data from memory
    workers:int = 8

    # Data augmentation parameters 
    # Determines the size of the image fed into the model
    resize_size:int = 256
    crop_size:int = 224

    # Optimizer parameters
    optimizer:str = "sgd"
    learning_rate:float = 0.256
    opt_momentum:float = 0.9
    opt_alpha:float = 0.9
    weight_decay:float = 0.0005

    # Scheduler parameters
    scheduler:str="steplr"
    lr_gamma:float = 0.97
    lr_min:float = 1e-5
    lr_warmup_epochs:int = 5
    lr_power:float = 4
    step_size:int=2

    # GPU settings (lightning)
    device="gpu"
    num_gpus=4

    # Logging settings
    save_dir = "./Logs"
    use_wandb = False
    project_name = "Classification"