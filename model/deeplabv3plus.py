import torch
import torch.nn as nn
from .backbone import Backbone
from .attention import ChannelAttention, SpatialAttention

# Set manual seed for reproducibility
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# DROPOUT_RATE = 0.2

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            dilation=dilation, 
            groups=in_channels, # Each input channel is convolved separately
            bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)
        

class AtrousConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_attention=False):
        super().__init__()

        self.atrous_conv = DepthwiseSeparableConv(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation
        )

        # Channel Attention module
        self.use_attention = use_attention
        if self.use_attention:
            self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        x = self.atrous_conv(x)

        if self.use_attention:
            x = self.ca(x)

        return x
    

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()

        # Dimensionality reduction (reduce channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        # Atrous convs that capture different scales
        self.atrous_conv_rate6 = AtrousConv(in_channels, out_channels, dilation=6, use_attention=use_attention)
        self.atrous_conv_rate12 = AtrousConv(in_channels, out_channels, dilation=12, use_attention=use_attention)
        self.atrous_conv_rate18 = AtrousConv(in_channels, out_channels, dilation=18, use_attention=use_attention)

        # Downsampling (reduce spatial dimensions)
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Projection layer (combine ASPP layers)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=DROPOUT_RATE)
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.atrous_conv_rate6(x)
        x3 = self.atrous_conv_rate12(x)
        x4 = self.atrous_conv_rate18(x)
        x5 = self.image_pooling(x)
        x5 = nn.functional.interpolate(x5, size=x1.size()[2:], mode="bilinear", align_corners=True)

        # Concatenate ASPP layers and apply projection
        concat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.project(concat)


class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_room_classes, num_icon_classes, num_heatmaps, use_attention=False):
        super().__init__()

        self.low_level_projection = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=DROPOUT_RATE)
        )

        # Spatial attention module
        self.use_attention = use_attention
        if self.use_attention:
            self.sa = SpatialAttention()

        # Shared decoder for segmentation and heatmap heads
        self.conv3x3 = nn.Sequential(
            DepthwiseSeparableConv(304, 256, kernel_size=3, padding=1), # 256 (ASPP output) + 48 (low-level features)
            # nn.Dropout2d(p=DROPOUT_RATE),
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
            # nn.Dropout2d(p=DROPOUT_RATE),
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1)
        )

        # Room segmentation head
        self.room_head = nn.Conv2d(256, num_room_classes, kernel_size=1)

        # Icon segmentation head
        self.icon_head = nn.Conv2d(256, num_icon_classes, kernel_size=1)

        # Heatmap regression head
        self.heatmap_head = nn.Conv2d(256, num_heatmaps, kernel_size=1)


    def forward(self, low_level_features, aspp_output, input_shape):
        low_level = self.low_level_projection(low_level_features)
        low_level_size = low_level.size()[2:]

        # First upsampling
        upsample1 = nn.functional.interpolate(
            aspp_output, 
            size=low_level_size, 
            mode="bilinear", 
            align_corners=True
        )
        
        # Spatial attention module
        if self.use_attention:
            upsample1 = self.sa(upsample1)
        
        # Concatenate high-level and low-level features
        concat = torch.cat([upsample1, low_level], dim=1)

        # Shared by segmentation and heatmap heads
        decoder_output = self.conv3x3(concat)

        # Room prediction
        room_output = nn.functional.interpolate(
            self.room_head(decoder_output), 
            size=input_shape, 
            mode="bilinear", 
            align_corners=True
        )

        # Icon prediction
        icon_output = nn.functional.interpolate(
            self.icon_head(decoder_output), 
            size=input_shape, 
            mode="bilinear", 
            align_corners=True
        )
        
        # Heatmap prediction
        heatmap_output = nn.functional.interpolate(
            self.heatmap_head(decoder_output), 
            size=input_shape, 
            mode="bilinear", 
            align_corners=True
        )
        
        # CubiCasa outputs (2 segmentation maps + 21 heatmaps)
        return room_output, icon_output, heatmap_output


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone="mobilenet_v2", attention=False, num_room_classes=12, num_icon_classes=11, num_heatmaps=21):
        super().__init__()

        self.backbone_name = backbone

        if attention:
            self.attention = 'ca_sa'
        else:
            self.attention = 'base'

        self.backbone = Backbone(backbone=backbone)

        self.aspp = ASPP(self.backbone.high_level_channels, 256, use_attention=attention)

        self.decoder = Decoder(
            self.backbone.low_level_channels, 
            num_room_classes, 
            num_icon_classes,
            num_heatmaps,
            use_attention=attention
        )

    def forward(self, x):
        input_shape = x.shape[2:] # Get Height x Width

        # Extract features from backbone
        low_level_features, high_level_features = self.backbone(x)

        # Pass high level features through ASPP        
        aspp_output = self.aspp(high_level_features)

        # Pass low level features and ASPP output through decoder
        room_output, icon_output, heatmap_output = self.decoder(low_level_features, aspp_output, input_shape)
        
        return room_output, icon_output, heatmap_output
    