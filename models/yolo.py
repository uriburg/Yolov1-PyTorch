import torch.nn as nn
import torchvision


class YOLOV1(nn.Module):
    r"""
    Model with three components:
    1. Backbone of resnet34 pretrained on 224x224 images from Imagenet
    2. 4 Conv,Batchnorm,LeakyReLU Layers for Yolo Detection Head
    3. Fc layers with final layer having S*S*(5B+C) output dimensions
        Final layer predicts [
            x_offset_box1,y_offset_box1,sqrt_w_box1,sqrt_h_box1,conf_box1, # box-1 params
            ...,
            x_offset_boxB,y_offset_boxB,sqrt_w_boxB,sqrt_h_boxB,conf_boxB, # box-B params
            p1, p2, ...., pC-1, pC  # class conditional probabilities
        ] for each S*S grid cell
    """
    def __init__(self, im_size, num_classes, model_config):
        super(YOLOV1, self).__init__()
        self.im_size = im_size
        self.im_channels = model_config['im_channels']
        self.backbone_channels = model_config['backbone_channels']
        self.yolo_conv_channels = model_config['yolo_conv_channels']
        self.backbone_spatial_size = model_config['backbone_spatial_size']
        self.leaky_relu_slope = model_config['leaky_relu_slope']
        self.yolo_fc_hidden_dim = model_config['fc_dim']
        self.yolo_fc_dropout_prob = model_config['fc_dropout']
        self.S = model_config['S']
        self.B = model_config['B']
        self.C = num_classes
        backbone = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )
        ###################
        # Backbone Layers #
        ###################
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        #########################
        # Detection Conv Layers #
        #########################
        self.conv_yolo_layers = nn.Sequential(
            nn.Conv2d(self.backbone_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Conv2d(self.yolo_conv_channels,
                      self.yolo_conv_channels,
                      3,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(self.yolo_conv_channels),
            nn.LeakyReLU(self.leaky_relu_slope)
            )

        #######################
        # Detection FC Layers #
        #######################
        self.fc_yolo_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone_spatial_size * self.backbone_spatial_size *
                      self.yolo_conv_channels,
                      self.yolo_fc_hidden_dim),
            nn.LeakyReLU(self.leaky_relu_slope),
            nn.Dropout(self.yolo_fc_dropout_prob),
            nn.Linear(self.yolo_fc_hidden_dim,
                      self.S * self.S * (5 * self.B + self.C)),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.conv_yolo_layers(out)
        out = self.fc_yolo_layers(out)
        return out
