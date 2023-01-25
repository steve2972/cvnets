import torch

from models.mobilevit_v2 import MobileVitv2
from models.mobilenet_v3 import MobileNetV3

from utils import count_params


test = torch.rand((1,3,224,224))
model = MobileVitv2(num_classes=1000)
print(model)

end_points = model.extract_end_points(test, True, True)
for k in end_points.keys():
    print(k, ':', end_points[k].shape)

print(model(test).shape)


print(f"Number of trainable parameters: {count_params(model)/1e6:.2f} M")