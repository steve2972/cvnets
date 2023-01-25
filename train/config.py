class Config(object):
    root:str = "/home/hyperai1/jhsong/Data/ImageNet"

    epochs:int = 150
    batch_size:int = 128
    workers:int = 8

    resize_size:int = 256
    crop_size:int = 224

    optimizer = "adamw"
    learning_rate = 0.3
    scheduler="steplr"

    device="gpu"
    num_gpus=4

    save_dir = "./Logs"
    use_wandb = False
    project_name = "Classification"