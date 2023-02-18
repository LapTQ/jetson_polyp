import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary


class Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), # do NOT use padding='same' for onnx
            nn.BatchNorm2d(mid_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Mish(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(inplace=True)
        )

    def forward(self, x):
        return self.sequential(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.MaxPool2d(3, stride=2, padding=1),
            Block(in_channels, out_channels)
        )

    def forward(self, x):
        return self.sequential(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_short = nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)
        self.conv = Block(in_channels, out_channels, in_channels // 2)

    def forward(self, x, x_short):
        x_short = self.conv_short(x_short)
        x = self.up(x)

        # ===== commect this block require x to be divisible by 32, 64,... =====
        # dy = x_short.size()[2] - x.size()[2]
        # dx = x_short.size()[3] - x.size()[3]
        # x = F.pad(x, [torch.div(dx, 2, rounding_mode='floor'),
        #               dx - torch.div(dx, 2, rounding_mode='floor'),
        #               torch.div(dy, 2, rounding_mode='floor'),
        #               dy - torch.div(dx, 2, rounding_mode='floor')])
        # ======================================================================

        x = torch.cat([x_short, x], dim=1)
        x = self.conv(x)
        return x


class Out(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inp = Block(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 256)
        self.up5 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up2 = Up(64, 16)
        self.up1 = Up(32, 16)
        self.out = Out(16, n_classes)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        assert x.size()[2] % 64 == 0 and x.size()[3] % 64 == 0, 'input shape must be divisible of 64'
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.down5(x5)
        x = self.up5(x, x5)
        x = self.up4(x, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)
        # x = self.softmax(x)

        return x


if __name__ == '__main__':
    net = UNet(3, 1)

    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    # net = deeplabv3_mobilenet_v3_large(pretrained=True)
    # net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)

    print(summary(net, (3, 448, 448)))

    a = torch.ones((4, 3, 448, 448))
    print(net(a).shape)  # add ['out'] for deeplab


