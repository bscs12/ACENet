import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from collections import OrderedDict


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "noact":
        module = SLU()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


class Upsample1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample1, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class Upsample2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=1)
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    """Residual layer with `in_channels` inputs."""
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""
    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)
    
    
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        m_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, m_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(m_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(m_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class SpatialTemporalFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialTemporalFusionModule, self).__init__()
        self.coordinate_attention = CoordinateAttention(in_channels=in_channels, out_channels=out_channels)
        self.conv0 = nn.Conv2d(in_channels * 3, out_channels, kernel_size=1)
        self.rconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.rrconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.rrbn = nn.BatchNorm2d(out_channels)
        self.rrrelu = nn.ReLU()

    def forward(self, x1, x2, x3):
        feat1 = x1 + x2
        feat2 = x1 + x3
        feat1 = self.coordinate_attention(feat1)
        feat2 = self.coordinate_attention(feat2)                
        feat3 = torch.cat((x1, feat1, feat2), dim=1)        
        feat3 = self.conv0(feat3)
        # n = self.rrbn(self.rrconv(self.rconv(feat3)))
        # output = self.rrrelu(feat3 + n)

        return feat3


class MultimodalFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultimodalFusionModule, self).__init__()
        self.conv0 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
               
        self.rconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.rrconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.rrbn = nn.BatchNorm2d(in_channels)
        self.rrrelu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=4, padding=4)
        
        self.conv = nn.Conv2d(5 * in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, rgb, t):
        feat1 = rgb + t
        feat1_rgb = rgb * feat1
        feat1_t = t * feat1
        
        feat2 = torch.cat((feat1_rgb, feat1_t), dim=1)
        feat2 = self.conv0(feat2)

        n = self.rrbn(self.rrconv(self.rconv(feat2)))
        feat2 = self.rrrelu(feat2 + n)
        
        feat2_1 = self.conv1(feat2)
        feat2_2 = self.conv2(feat2)
        feat2_3 = self.conv3(feat2)
        feat2_4 = self.conv4(feat2)
        
        feat3 = torch.cat((feat2, feat2_1, feat2_2, feat2_3, feat2_4), dim=1)
        output = self.conv(feat3)

        return output


class MyConv(nn.Module):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MyConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode

    def forward(self, x):
        num_channels = x.size(1)  # 获取输入张量的通道数
        conv_layer = nn.Conv2d(num_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                               self.dilation, self.groups, self.bias, self.padding_mode)
        output = conv_layer(x)
        return output


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dilation, groups=1, reduction=0.0625,
        kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.field_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1)
            self.func_spatial = self.get_spatial_attention

        self.field_fc = nn.Conv2d(attention_channel, self.field_size * self.field_size, 1)
        self.func_field = self.get_field_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size, 1, 1)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention
    
    def get_field_attention(self, x):
        field_attention = self.field_fc(x).view(x.size(0), 1, 1, 1, 1, 1, self.field_size, self.field_size)
        field_attention = torch.sigmoid(field_attention / self.temperature)
        return field_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_field(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.field_size = (kernel_size + (kernel_size - 1) * (dilation - 1))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
            reduction=reduction, kernel_num=kernel_num, dilation=dilation)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups,
            self.kernel_size, self.kernel_size, self.field_size, self.field_size, requires_grad=True))

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common
            
        self.output_conv = MyConv(out_channels=64, kernel_size=1, stride=1, padding=0)

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, field_attention, kernel_attention= self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        # print(spatial_attention.shape,field_attention.shape)
        aggregate_weight = spatial_attention * field_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        # print(aggregate_weight.shape)
        aggregate_weight = torch.sum(aggregate_weight, dim=(6, 7))
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        # print(x.shape,aggregate_weight.shape)
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups * batch_size)
        # print(output.shape)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        # output = self.output_conv(output)
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


class AdaptiveContextExtractModule(nn.Module):
    def __init__(self):
        super(AdaptiveContextExtractModule, self).__init__()
        self.epsilon = 1e-4
        self.swish = h_swish()

        self.conv0 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=3, padding=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=5, padding=5)

        self.context_w = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.context_w_relu = nn.ReLU()

        self.context_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        context_w = self.context_w_relu(self.context_w)
        context_w = context_w / (torch.sum(context_w, dim=0) + self.epsilon)
        output = self.context_conv(self.swish(
            context_w[0] * x +
            context_w[1] * x1 +
            context_w[2] * x2 +
            context_w[3] * x3
        ))
        output = self.sigmoid(output)
        return output