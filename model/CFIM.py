import torch
from torch import nn
class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // 4, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)
class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class CFIM(nn.Module):
    def __init__(self, channel):
        super(CFIM, self).__init__()

        self.att_c1 = ChannelAttention(channel * 3)
        self.att_s = SpatialAttention()

        self.fc_i1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fc_t1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fc_i2 = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.fc_t2 = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.transfer = TransBasicConv2d(32, 32, kernel_size=2, stride=2,padding=0, dilation=1, bias=False)

        self.conv_upsample1 = ConvBR(channel, channel, 3, padding=1)
        self.conv_out1 = ConvBR(channel, channel,1)
        self.conv_out2 = ConvBR(channel, channel,1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, i, t, f=None):
        t = self.conv_upsample1(self.upsample(t))
        sa_i = i.mul(self.att_s(i))
        sa_t = t.mul(self.att_s(t))
        i1 = self.fc_i1(i)
        t1 = self.fc_t1(t)

        mix2 = torch.cat([torch.cat([i1, torch.mul(i1, t1)], dim=1), t1], dim=1)
        ca = mix2.mul(self.att_c1(mix2))

        atti = self.fc_i2(ca) + sa_t
        attt = self.fc_t2(ca) + sa_i
        out_att = self.conv_out1(atti + attt)
        if f is not None:
            f = self.transfer(f)
            out_att = (f * out_att) + out_att
            out_att = self.conv_out2(out_att)
        return out_att
