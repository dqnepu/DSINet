from model.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
from model.CFIM import CFIM
from model.EfficientNet import EfficientNet
from model.FRM import FRM
from model.MEM import MEM
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


class NCD(nn.Module):
    def __init__(self, channel):
        super(NCD, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (self.conv_upsample2(self.upsample(self.upsample(x1))) *
                self.conv_upsample3(self.upsample(x2)) * x3)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class DSINet(nn.Module):
    def __init__(self, channel=32):
        super(DSINet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.model = EfficientNet.from_pretrained(f'efficientnet-b{7}', advprop=True)

        self.frm1 = FRM(channel)
        self.frm2 = FRM(channel)
        self.frm3 = FRM(channel)
        self.frm4 = FRM(channel)

        def channel_c(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, channel, kernel_size=1),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )

        self.ChannelNormalization_1 = channel_c(64)
        self.ChannelNormalization_2 = channel_c(128)
        self.ChannelNormalization_3 = channel_c(320)
        self.ChannelNormalization_4 = channel_c(512)
        self.ChannelNormalization_5 = channel_c(48)
        self.ChannelNormalization_6 = channel_c(80)
        self.ChannelNormalization_7 = channel_c(224)
        self.ChannelNormalization_8 = channel_c(640)

        self.rfe1 = MEM(channel,channel)
        self.rfe2 = MEM(channel,channel)
        self.rfe3 = MEM(channel,channel)
        self.rfe4 = MEM(channel,channel)

        self.ncd = NCD(channel)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

        self.cfim1 = CFIM(channel)
        self.cfim2 = CFIM(channel)
        self.cfim3 = CFIM(channel)
    def forward(self, x):
        x_eff = self.model.initial_conv(x)
        features = self.model.get_blocks(x_eff)
        x_sal1 = self.ChannelNormalization_5(features[0])
        x_sal2 = self.ChannelNormalization_6(features[1])
        x_sal3 = self.ChannelNormalization_7(features[2])
        x_sal4 = self.ChannelNormalization_8(features[3])
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        x1_nor = self.ChannelNormalization_1(x1) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2) # 32x22x22
        x3_nor = self.ChannelNormalization_3(x3) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4) # 32x11x11

        x_sal1_rfe = self.rfe1(x_sal1)
        x_sal2_rfe = self.rfe2(x_sal2)
        x_sal3_rfe = self.rfe3(x_sal3)
        x_sal4_rfe = self.rfe4(x_sal4)

        f_tv1 = self.frm1(x1_nor, x_sal1_rfe)
        f_tv2 = self.frm2(x2_nor, x_sal2_rfe)
        f_tv3 = self.frm3(x3_nor, x_sal3_rfe)
        f_tv4 = self.frm4(x4_nor, x_sal4_rfe)

        f_ifm1 = self.cfim1(f_tv3, f_tv4)
        f_ifm2 = self.cfim2(f_tv2, f_tv3, f_ifm1)
        f_ifm3 = self.cfim3(f_tv1, f_tv2, f_ifm2)

        prediction = self.upsample_4(self.ncd(f_ifm1, f_ifm2, f_ifm3))

        return prediction, self.sigmoid(prediction)