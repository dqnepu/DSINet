import torch
from torch import nn

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Channel_Shuffle(nn.Module):
    def __init__(self, num_groups):
        super(Channel_Shuffle, self).__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.FloatTensor):
        batch_size, chs, h, w = x.shape
        chs_per_group = chs // self.num_groups
        x = x.view(batch_size, self.num_groups, chs_per_group, h, w).transpose(1, 2).contiguous()
        return x.view(batch_size, -1, h, w)

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(channel // ratio, channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = self.conv1(max_out)
        return self.sigmoid(x)

class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        max_out, _ = torch.max(x2, dim=1, keepdim=True)
        avg_out = torch.mean(x2, dim=1, keepdim=True)
        x2 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x2)
        return self.sigmoid(x2) * x1

class Gate(nn.Module):
    def __init__(self, in_channel, ratio, out_channel):
        super(Gate, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(out_channel, out_channel // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(out_channel // ratio, out_channel, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = max_out + avg_out
        return x * self.sigmoid(out)

class SSU(nn.Module):
    def __init__(self, in_channel, ratio):
        super(SSU, self).__init__()
        self.gate1 = Gate(in_channel, ratio, out_channel=32)
        self.gate2 = Gate(in_channel, ratio, out_channel=32)
        self.att1 = SA()
        self.att2 = SA()
        self.out_conv = BasicConv2d(in_channel * 2, in_channel, 1, 1)

    def forward(self, x_pvt, x_eff, gate):
        x_pvt = self.gate1(x_pvt)
        x_eff = self.gate2(x_eff)
        feat_1 = self.att1(x_pvt, x_eff)
        feat_2 = self.att2(x_eff, x_pvt)
        out1 = x_pvt + gate * feat_1
        out2 = x_eff + gate * feat_2
        return self.out_conv(torch.cat([out1, out2], dim=1))

class FRM(nn.Module):
    def __init__(self, in_channel):
        super(FRM, self).__init__()
        self.channel_shuffle_0 = Channel_Shuffle(num_groups=4)
        self.ssu = SSU(in_channel, 4)
        self.avg_1 = nn.AdaptiveAvgPool2d(1)
        self.avg_2 = nn.AdaptiveAvgPool2d(1)
        self.max_1 = nn.AdaptiveMaxPool2d(1)
        self.max_2 = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Sigmoid()
        )
        self.spatial_att_1 = SpatialAttention()
        self.spatial_att_2 = SpatialAttention()
        self.channel_att_1 = ChannelAttention(in_channel)
        self.channel_att_2 = ChannelAttention(in_channel)
        self.basic_conv1 = BasicConv2d(in_channel, in_channel, 3, stride=1, padding=1)
        self.basic_conv2 = BasicConv2d(in_channel, in_channel, 1, stride=1, padding=0)

    def forward(self, x_pvt, x_eff):
        x1_chashu = self.channel_shuffle_0(x_pvt)
        x2_chashu = self.channel_shuffle_0(x_eff)
        #SSU
        bs = x_eff.shape[0]
        eff_pool = self.avg_1(x_eff) + self.max_1(x_eff)
        eff_pool = eff_pool.view(bs, -1)
        pvt_pool = self.avg_2(x_pvt) + self.max_2(x_pvt)
        pvt_pool = pvt_pool.view(bs, -1)
        feat = torch.cat((eff_pool, pvt_pool), dim=1)
        gate = self.fc(feat).mean(dim=-1).view(bs, 1, 1, 1)
        ssu_out = self.ssu(x_pvt, x_eff, gate)
        #FCU
        pvt_1 = x1_chashu * self.spatial_att_1(x2_chashu)
        eff_1 = x2_chashu * self.spatial_att_2(x1_chashu)
        pvt_1 = pvt_1 * self.channel_att_1(pvt_1) + x1_chashu
        eff_1 = eff_1 * self.channel_att_2(eff_1) + x2_chashu
        fcu_out = pvt_1 + eff_1
        
        return self.basic_conv2(self.basic_conv1(ssu_out * fcu_out))