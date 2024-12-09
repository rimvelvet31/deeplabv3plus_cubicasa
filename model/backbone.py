import torch.nn as nn
import torchvision
from torchvision.models import (mobilenet_v2, MobileNet_V2_Weights,
                                mobilenet_v3_large, MobileNet_V3_Large_Weights,
                                resnet50, ResNet50_Weights,
                                efficientnet_b2, EfficientNet_B2_Weights,
                                efficientnet_v2_s, EfficientNet_V2_S_Weights)

class Backbone(nn.Module):
    def __init__(self, backbone="mobilenet_v2"):
        super().__init__()

        self.backbone_name = backbone

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

        elif backbone == "efficientnet_b2":
            self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
            self.low_level_channels = 24
            self.high_level_channels = 1408
        
        elif backbone == "efficientnet_v2_s":
            self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            self.low_level_channels = 24
            self.high_level_channels = 1280

        else:
            raise ValueError("Unsupported backbone")


    def forward(self, x):
        # ResNet-50
        if self.backbone_name == "resnet50":
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            low_level_features = self.backbone.layer1(x)

            x = self.backbone.layer2(low_level_features)
            x = self.backbone.layer3(x)
            high_level_features = self.backbone.layer4(x)

        # EfficientNet-B2
        elif self.backbone_name == "efficientnet_b2":
            features = self.backbone.features
            low_level_features = features[:3](x)
            high_level_features = features[3:](low_level_features)

        # EfficientNetV2-S
        elif self.backbone_name == "efficientnet_v2_s":
            features = self.backbone.features
            low_level_features = features[:2](x)
            high_level_features = features[2:](low_level_features)

        # MobileNetV3
        elif isinstance(self.backbone, torchvision.models.MobileNetV3):
            low_level_features = self.backbone.features[:5](x)
            high_level_features = self.backbone.features[5:](low_level_features)

        # MobileNetV2
        else:
            features = self.backbone.features
            low_level_features = features[:4](x)
            high_level_features = features[4:](low_level_features)

        return low_level_features, high_level_features
