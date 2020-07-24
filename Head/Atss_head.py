import torch
from torch import nn
import math
from torch.nn import functional as F

class ATSSHead(nn.Module):
    def __init__(self, in_channels, n_class, n_conv, prior, regression_type):
        super(ATSSHead, self).__init__()
        num_classes = n_class - 1
        num_anchors = 1

        self.regression_type = regression_type

        cls_tower = []
        bbox_tower = []
        for i in range(n_conv):
            # if self.cfg.MODEL.ATSS.USE_DCN_IN_TOWER and i == n_conv - 1:
            #     conv_func = DFConv2d
            # else:
            conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.BatchNorm2d(num_features=in_channels,momentum=0.01, eps=1e-3))

            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.BatchNorm2d(num_features=in_channels, momentum=0.01, eps=1e-3))
            # bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, num_anchors * 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = prior
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        if regression_type == 'POINT':
            assert num_anchors == 1, "regressing from a point only support num_anchors == 1"
            torch.nn.init.constant_(self.bbox_pred.bias, 4)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.regression_type == 'POINT':
                bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

            centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale