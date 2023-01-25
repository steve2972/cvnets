from torch import nn, Tensor

from typing import Optional, Dict

"""Adapted from https://github.com/apple/ml-cvnets"""

class BaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = None
        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.conv_1x1_exp = None
        self.classifier = None
        self.round_nearest = 8

    def check_model(self):
        assert self.conv_1 is not None, "Please implement self.conv_1"
        assert self.layer_1 is not None, "Please implement self.layer_1"
        assert self.layer_2 is not None, "Please implement self.layer_2"
        assert self.layer_3 is not None, "Please implement self.layer_3"
        assert self.layer_4 is not None, "Please implement self.layer_4"
        assert self.layer_5 is not None, "Please implement self.layer_5"
        assert self.conv_1x1_exp is not None, "Please implement self.conv_1x1_exp"
        assert self.classifier is not None, "Please implement self.classifier"

    def update_classifier(self, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        if isinstance(self.classifier, nn.Sequential):
            in_features = self.classifier[-1].in_features
            layer = nn.Linear(in_features, n_classes, bias=True)
            self.classifier[-1] = layer
        else:
            in_features = self.classifier.in_features
            layer = nn.Linear(in_features, n_classes, bias=True)

            head_init_scale = 0.001
            layer.weight.data.mul_(head_init_scale)
            layer.bias.data.mul_(head_init_scale)

            self.classifier = layer

    def extract_end_points(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
    ) -> Dict[str, Tensor]:
        out_dict = {"out_input": x}

        x = self.conv_1(x) # 112 x112
        out_dict["out_conv"] = x

        x = self.layer_1(x)  # 112 x112
        out_dict["out_l1"] = x

        x = self.layer_2(x)  # 56 x 56
        out_dict["out_l2"] = x

        x = self.layer_3(x) # 28 x 28
        out_dict["out_l3"] = x

        x = self.layer_4(x)  # 14 x 14
        out_dict["out_l4"] = x

        if use_l5:
            x = self.layer_5(x)  # 7 x 7
            out_dict["out_l5"] = x
            if use_l5_exp:
                x = self.conv_1x1_exp(x)
                out_dict["out_l5_exp"] = x
        return out_dict

    def _extract_features(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self._extract_features(x)
        x = self.classifier(x)
        return x