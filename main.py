from train import train_model, Config
from models import MobileVitv2, MobileNetV3

model = MobileNetV3(mv_type="large")
train_model(model, Config)