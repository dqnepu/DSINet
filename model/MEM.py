import torch
import torch.nn as nn
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)

class MEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MEM, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, in_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel//4, out_channel//4, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel//4, out_channel//4, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel//4, out_channel//4, 3, padding=7, dilation=7)
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channel//4, out_channel//4, kernel_size=(9, 1), padding=(4, 0)),
            BasicConv2d(out_channel//4, out_channel//4, 3, padding=9, dilation=9)
        )
        self.group1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, groups=4,
                             kernel_size=3, stride=1, bias=False, padding=1)
        self.group2 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, groups=4,
                             kernel_size=3, stride=1, bias=False, padding=1)
        self.sa = SpatialAttention()
        self.conv_cat_1 = BasicConv2d(in_channel, in_channel, 1)
        self.conv_cat_3 = BasicConv2d(in_channel, in_channel, 3, padding=1)
    def forward(self, x):
        # x1, x2, x3, x4 = torch.split(x, split_size_or_sections=4, dim=1)
        x_sa = self.sa(x) * x
        x1, x2, x3, x4 = torch.chunk(x_sa, chunks=4, dim=1)
        x0 = self.branch0(x_sa)  #1*1卷积

        x1_d = self.branch1(x1 + x2)  #33卷积

        x2_d = self.branch2(x2 + x3 + x1_d)#55卷积

        x3_d = self.branch3(x3 + x4 + x2_d)#77卷积

        x4_d = self.branch4(x4 + x3_d)#99卷积

        x_d_cat = torch.cat([x1_d, x2_d, x3_d, x4_d], dim=1)

        x_12_group = self.group1(x_d_cat)
        # x_34_group = self.group2(x_34)
        x_out_0 = x_12_group + x0

        x_out = self.conv_cat_3(x_out_0)
        return x_out
