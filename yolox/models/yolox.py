import torch
import torch.nn as nn
import cv2
import numpy

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(7)

        self.backbone = backbone
        self.head = head
        
    def forward(
        self,
        x,
        x1=torch.zeros(1, 3, 64, 64),
        x2=torch.zeros(1, 3, 64, 64),
        y=torch.zeros(1, 3, 64, 64),
        y1=torch.zeros(1, 3, 64, 64),
        y2=torch.zeros(1, 3, 64, 64),
        targets=None,
    ):
        fpn_outs = self.backbone(x, x1, x2, y, y1, y2)
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs