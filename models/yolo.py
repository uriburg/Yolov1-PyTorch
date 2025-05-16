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
    def __init__(self, im_size : int, num_classes : int, S : int, B : int):
        super(YOLOV1, self).__init__()
        self.im_size = im_size
        self.S = S
        self.B = B
        self.C = num_classes

        self.backbone = nn.Sequential(
            Conv(3, 8, 3, 1, 1),
            nn.MaxPool2d(2, 2, 0),
            Conv(8, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2, 0),
            Conv(16, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2, 0),
            Conv(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2, 0), )

        self.head = nn.Conv2d(64, 5 * self.B + self.C, 1)

    def forward(self, x):
        out = self.head(self.backbone(x))
        # Reshape conv output to Batch x S x S x (5B+C)
        out = out.permute(0, 2, 3, 1)
        return out


class Conv(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))