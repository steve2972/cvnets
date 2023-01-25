from torch import nn

from models import BaseModel
from layers import ConvBlock
from layers.modules.mobilenet import InvertedResidualSE
from utils.math_utils import make_divisible

class MobileNetV3(BaseModel):
    def __init__(self, num_classes:int=1000, mv_type:str="small") -> None:
        super().__init__()
        mv_avail = {"large", "small"}
        mv_type = mv_type.lower()
        if mv_type not in mv_avail:
            raise ValueError(f"Available MobileNetV3 types: {mv_avail}. Receieved {mv_type}")
        if mv_type == 'small':
            layers = MobileNetV3_Small
        else:
            layers = MobileNetV3_Large

        self.conv_1 = ConvBlock(3, 16, 3, strides=2, activation='hswish', padding=1)
        self.layer_1 = self._make_layer(layers["layer_1"])
        self.layer_2 = self._make_layer(layers["layer_2"])
        self.layer_3 = self._make_layer(layers["layer_3"])
        self.layer_4 = self._make_layer(layers["layer_4"])
        self.layer_5 = self._make_layer(layers["layer_5"])

        features = layers["layer_5"][-1][2]
        last_features = features*6
        last_point_features = make_divisible(last_features)
        self.conv_1x1_exp = ConvBlock(features, last_features, 1, activation='hswish')

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(last_features, last_point_features, 1, padding='same'),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Conv2d(last_point_features, num_classes, 1, padding='same'),
            nn.Flatten()
        )

    def _make_layer(self,mv3_config):
        mv3_block = []

        for expansion, in_features, features, kernel_size, stride, se_ratio, activation in mv3_config:
            layer = InvertedResidualSE(
                in_features=in_features,
                out_features=features,
                stride=stride,
                expand_ratio=expansion,
                kernel_size=kernel_size,
                activation=activation,
                use_se=se_ratio
            )
            mv3_block.append(layer)

        return nn.Sequential(*mv3_block)

MobileNetV3_Small = {
    "layer_1": [[1, 16, 16, 3, 2, True, "relu"]],
    "layer_2": [[4.5, 16, 24, 3, 2, False, "relu"]],
    "layer_3":[[3.67, 24, 24, 3, 1, False, "relu"]],
    "layer_4": [
        [4, 24, 40, 5, 2, True, "hswish"],
        [6, 40, 40, 5, 1, True, "hswish"],
        [6, 40, 40, 5, 1, True, "hswish"],
        [3, 40, 48, 5, 1, True, "hswish"],
        [3, 48, 48, 5, 1, True, "hswish"],
    ],
    "layer_5": [
        [6, 48, 96, 5, 2, True, "hswish"],
        [6, 96, 96, 5, 1, True, "hswish"],
        [6, 96, 96, 5, 1, True, "hswish"],
    ]
}

MobileNetV3_Large = {
    "layer_1": [[1, 16, 16, 3, 1, False, "relu"]],
    "layer_2": [
        [4, 16, 24, 3, 2, False, "relu"],
        [3, 24, 24, 3, 1, False, "relu"],
    ],
    "layer_3":[
        [3, 24, 40, 5, 2, True, "relu"],
        [3, 40, 40, 5, 1, True, "relu"],
        [3, 40, 40, 5, 1, True, "relu"]
    ],
    "layer_4": [
        [6, 40, 80, 3, 2, False, "hswish"],
        [2.5, 80, 80, 3, 1, False, "hswish"],
        [2.3, 80, 80, 3, 1, False, "hswish"],
        [2.3, 80, 80, 3, 1, False, "hswish"],
        [6, 80, 112, 3, 1, True, "hswish"],
        [6, 112, 112, 3, 1, True, "hswish"],

    ],
    "layer_5": [
        [6, 112, 160, 5, 2, True, "hswish"],
        [6, 160, 160, 5, 1, True, "hswish"],
        [6, 160, 160, 5, 1, True, "hswish"],
    ]
}