import torch.nn.functional as F
from torch import nn
import torch
import cv2
from .network_blocks import BaseConv, CSPLayer, DWConv, ResLayer, SPPBottleneck


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark2", "dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)           # wid_mul=0.5   base_channels=32
        base_depth = max(round(dep_mul * 3), 1)     # dep_mul=0.33  base_depth=1

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act="silu"),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        
    def forward(self, x):    # [1, 32, 320, 320]
        outputs = {}

        x = self.dark2(x)
        outputs["dark2"] = x # [1, 64, 160, 160]

        x = self.dark3(x)
        outputs["dark3"] = x # [1, 128, 80, 80]

        x = self.dark4(x)
        outputs["dark4"] = x # [1, 256, 40, 40]
        
        x = self.dark5(x)
        outputs["dark5"] = x # [1, 512, 20, 20]
        
        return {k: v for k, v in outputs.items() if k in self.out_features}