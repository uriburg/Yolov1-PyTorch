import torch.nn as nn


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
    def __init__(self, im_size : int, num_classes : int, S : int, B : int, im_channels : int, use_conv : bool, shrink_network = 1):
        super(YOLOV1, self).__init__()
        self.im_size = im_size
        self.S = S
        self.B = B
        self.C = num_classes
        self.im_channels = im_channels
        self.output_dim = self.S * self.S * (self.B * 5 + self.C)
        self.use_conv = use_conv

        self.backbone = nn.Sequential(Conv(self.im_channels, 16, 3, 1, 1, False),
                                      nn.MaxPool2d(2, 2, 0),
                                      Conv(16, 16, 3, 1, 1, True),
                                      Conv(16, 16, 3, 1, 1, True),
                                      nn.MaxPool2d(2, 2, 0),
                                      Conv(16, 32, 3, 1, 1, True),
                                      Conv(32, 32, 3, 1, 1, True),
                                      nn.MaxPool2d(2, 2, 0),
                                      Conv(32, 64, 3, 1, 1, True),
                                      Conv(64, 64, 3, 1, 1, True),
                                      nn.MaxPool2d(2, 2, 0),
                                      Conv(64, 64, 3, 1, 1, True),
                                      Conv(64, 128 , 3, 1, 1, True),
                                      nn.MaxPool2d(2, 2,0),
                                      Conv(128, 128, 3, 1, 1, True),
                                      )

        if self.use_conv:
            self.head = nn.Conv2d(128, 5 * self.B + self.C, 1)
        else:
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(((im_size // (2 ** 5)) ** 2) * 128, self.output_dim))

        # Another architecture need to be tested
        # self.backbone = nn.Sequential(Conv(3, 64, 4, 2, 0, False),
        #                           nn.MaxPool2d(2, 2, 0),
        #                           Conv(64, 128, 3, 1, 0, True),
        #                           Conv(128, 128, 3, 1, 1, True),
        #                           Conv(128, 128, 3, 1, 0, True),
        #                           nn.MaxPool2d(2, 2, 0),
        #                           Conv(128, 128, 3, 1, 1, True),
        #                           Conv(128, 64, 3, 1, 0, True),
        #                           Conv(64, 64, 3, 1, 1, True),
        #                           Conv(64, 64, 3, 1, 0, True),
        #                           nn.MaxPool2d(2, 2, 0),
        #                           )
        #
        # self.head = nn.Sequential(nn.Flatten(),
        #                           nn.Linear(1024, 1024),
        #                           nn.Linear(1024, self.output_dim),
        #                           )
    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        if self.use_conv:
            # Reshape conv output to Batch x S x S x (5B+C)
            out = out.permute(0, 2, 3, 1)
        out = out.view(-1, self.S, self.S, self.B * 5 + self.C)
        return out


class Conv(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, depthwise=False):
        super(Conv, self).__init__()
        if not depthwise:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True),)
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU(inplace=True),)

    def forward(self, x):
        return self.conv(x)