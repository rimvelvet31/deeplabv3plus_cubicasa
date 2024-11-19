import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v2, mobilenet_v3_large, resnet50, MobileNet_V2_Weights, MobileNet_V3_Large_Weights, ResNet50_Weights


class Backbone(nn.Module):
    def __init__(self, backbone="mobilenet_v2"):
        super().__init__()

        if backbone == "mobilenet_v2":
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.low_level_channels = 24
            self.high_level_channels = 1280

        elif backbone == "mobilenet_v3":
            self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
            self.low_level_channels = 40
            self.high_level_channels = 960

        elif backbone == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.low_level_channels = 256
            self.high_level_channels = 2048

        else:
            raise ValueError("Unsupported backbone: Choose either 'mobilenet_v2', 'mobilenet_v3', or 'resnet50'")

    def forward(self, x):
        if isinstance(self.backbone, torchvision.models.ResNet):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            # Low level features
            low_level_features = self.backbone.layer1(x)

            # High level features
            x = self.backbone.layer2(low_level_features)
            x = self.backbone.layer3(x)
            high_level_features = self.backbone.layer4(x)

        elif isinstance(self.backbone, torchvision.models.MobileNetV3):
            low_level_features = self.backbone.features[:5](x)
            high_level_features = self.backbone.features[5:](low_level_features)

        else:
            features = self.backbone.features
            low_level_features = features[:4](x)
            high_level_features = features[4:](low_level_features)

        return low_level_features, high_level_features
