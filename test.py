import torch

from models.mobilenet_v3 import MobileNetV3



test = torch.rand((1,3,224,224))
model = MobileNetV3(mv_type="large")

