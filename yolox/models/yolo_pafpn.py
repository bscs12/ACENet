import scipy.misc
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from .darknet import CSPDarknet
from .network_blocks import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark2", "dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        epsilon=1e-4
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        self.swish = h_swish()
        self.epsilon = epsilon
        Conv = DWConv if depthwise else BaseConv

        self.spatial_temporal_fusion = SpatialTemporalFusionModule(in_channels=3, out_channels=3)
        
        self.multimodal_fusion = MultimodalFusionModule(in_channels=3, out_channels=32)

        self.p2_in_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.p3_in_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.p4_in_conv = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.p5_in_conv = nn.Conv2d(512, 64, kernel_size=1, stride=1)

        self.conv0 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.ODConv2d1 = ODConv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1,
            dilation=1, groups=1, reduction=0.0625, kernel_num=4)
        self.ODConv2d2 = ODConv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=3,
            dilation=3, groups=1, reduction=0.0625, kernel_num=4)
        self.ODConv2d3 = ODConv2d(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=5,
            dilation=5, groups=1, reduction=0.0625, kernel_num=4)

        self.context_w = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.context_w_relu = nn.ReLU()
        self.context_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        self.adaptive_context_extract = AdaptiveContextExtractModule()
        
        # 可学习权重
        self.p4_td_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_td_w_relu = nn.ReLU()
        self.p3_td_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_td_w_relu = nn.ReLU()
        self.p2_out_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_out_w_relu = nn.ReLU()
        self.p3_out_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_out_w_relu = nn.ReLU()
        self.p4_out_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_out_w_relu = nn.ReLU()
        self.p5_out_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_out_w_relu = nn.ReLU()

        # 上采样
        self.p5_in_upsample1 = Upsample1(64, 64)
        self.p4_td_conv = conv2d(64, 64, kernel_size=3)
        self.p4_td_upsample = Upsample1(64, 64)
        self.p3_td_conv = conv2d(64, 64, kernel_size=3)
        self.p3_td_upsample = Upsample1(64, 64)
        self.p2_out_conv = conv2d(64, 64, kernel_size=3)

        # 下采样
        self.p2_out_downsample = conv2d(64, 64, kernel_size=3, stride=2)
        self.p3_out_conv = conv2d(64, 64, kernel_size=3)
        self.p3_out_downsample = conv2d(64, 64, kernel_size=3, stride=2)
        self.p4_out_conv = conv2d(64, 64, kernel_size=3)
        self.p4_out_downsample = conv2d(64, 64, kernel_size=3, stride=2)
        self.p5_out_conv = conv2d(64, 64, kernel_size=3)

        self.output1_conv = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.output2_conv = nn.Conv2d(64, 256, kernel_size=3, stride=2)
        self.output3_conv = nn.Conv2d(64, 512, kernel_size=3, stride=2)

    def forward(
        self, 
        input,
        input1=torch.zeros(1, 3, 64, 64),
        input2=torch.zeros(1, 3, 64, 64),
        tinput=torch.zeros(1, 3, 64, 64),
        tinput1=torch.zeros(1, 3, 64, 64),
        tinput2=torch.zeros(1, 3, 64, 64),
    ):
        # 时空融合
        input_stf = self.spatial_temporal_fusion(input, input1, input2)
        tinput_stf = self.spatial_temporal_fusion(tinput, tinput1, tinput2)
        
        # 多模态融合
        input_mf = self.multimodal_fusion(input_stf, tinput_stf)

        # backbone
        out_features = self.backbone(input_mf)       
        features = [out_features[f] for f in self.in_features]
        [feat2, feat3, feat4, feat5] = features

        # FPN的输入
        p2_in, p3_in, p4_in, p5_in = feat2, feat3, feat4, feat5
        
        # 减少通道数
        p2_in = self.p2_in_conv(p2_in)
        p3_in = self.p3_in_conv(p3_in)
        p4_in = self.p4_in_conv(p4_in)
        p5_in = self.p5_in_conv(p5_in)

        # 自适应上下文提取
        p2_conv0 = self.conv0(p2_in)
        p2_ODConv2d1 = self.ODConv2d1(p2_in)
        p2_ODConv2d2 = self.ODConv2d2(p2_in)
        p2_ODConv2d3 = self.ODConv2d3(p2_in)
        context_w = self.context_w_relu(self.context_w)
        context_w = context_w / (torch.sum(context_w, dim=0) + self.epsilon)
        context = self.context_conv(self.swish(
            context_w[0] * p2_conv0 +
            context_w[1] * p2_ODConv2d1 +
            context_w[2] * p2_ODConv2d2 +
            context_w[3] * p2_ODConv2d3
        ))
        context = self.sigmoid(context)
        # context = self.adaptive_context_extract(p2_in)
        context1 = F.interpolate(context, scale_factor=1, mode='bilinear')
        context2 = F.interpolate(context, scale_factor=0.5, mode='bilinear')
        context3 = F.interpolate(context, scale_factor=0.25, mode='bilinear')
        context4 = F.interpolate(context, scale_factor=0.125, mode='bilinear')
        
        # 简单的注意力机制，用于确定更关注p4_in还是p5_in
        p4_td_w = self.p4_td_w_relu(self.p4_td_w)
        weight = p4_td_w / (torch.sum(p4_td_w, dim=0) + self.epsilon)
        p4_td = self.p4_td_conv(self.swish(weight[0] * p4_in + weight[1] * self.p5_in_upsample1(p5_in)))

        # 简单的注意力机制，用于确定更关注p3_in还是p4_td
        p3_td_w = self.p3_td_w_relu(self.p3_td_w)
        weight = p3_td_w / (torch.sum(p3_td_w, dim=0) + self.epsilon)
        p3_td = self.p3_td_conv(self.swish(weight[0] * p3_in + weight[1] * self.p4_td_upsample(p4_td)))

        # 简单的注意力机制，用于确定更关注p2_in还是p3_td
        p2_out_w = self.p2_out_w_relu(self.p2_out_w)
        weight = p2_out_w / (torch.sum(p2_out_w, dim=0) + self.epsilon)
        p2_out = self.p2_out_conv(self.swish(weight[0] * p2_in + weight[1] * self.p3_td_upsample(p3_td)))
        p2_out = p2_out + p2_out * context1

        # 简单的注意力机制，用于确定更关注p3_in还是p3_td还是p2_out
        p3_out_w = self.p3_out_w_relu(self.p3_out_w)
        weight = p3_out_w / (torch.sum(p3_out_w, dim=0) + self.epsilon)
        p3_out = self.p3_out_conv(
            self.swish(weight[0] * p3_in + weight[1] * p3_td + weight[2] * self.p2_out_downsample(p2_out)))
        p3_out = p3_out + p3_out * context2

        # 简单的注意力机制，用于确定更关注p4_in还是p4_td还是p3_out
        p4_out_w = self.p4_out_w_relu(self.p4_out_w)
        weight = p4_out_w / (torch.sum(p4_out_w, dim=0) + self.epsilon)
        p4_out = self.p4_out_conv(
            self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p3_out_downsample(p3_out)))
        p4_out = p4_out + p4_out * context3

        # 简单的注意力机制，用于确定更关注p5_in还是p4_out
        p5_out_w = self.p5_out_w_relu(self.p5_out_w)
        weight = p5_out_w / (torch.sum(p5_out_w, dim=0) + self.epsilon)
        p5_out = self.p5_out_conv(
            self.swish(weight[0] * p5_in + weight[1] * self.p4_out_downsample(p4_out)))
        p5_out = p5_out + p5_out * context4

        # 增加通道数，缩小分辨率
        p3_out = self.output1_conv(p3_out)
        p4_out = self.output2_conv(p4_out)
        p5_out = self.output3_conv(p5_out)
        
        outputs = (p3_out, p4_out, p5_out)
        return outputs