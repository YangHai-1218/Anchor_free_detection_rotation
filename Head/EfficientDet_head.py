import torch
from torch import nn
from utils.base_conv import SeparableConvBlock
from utils.Activation import Swish,MemoryEfficientSwish
import math


class EfficientDetHead(nn.Module):

    def __init__(self,compound_coef,prior,regression_type,num_anchors=1,num_classes=81,with_centerness=True,):
        super(EfficientDetHead, self).__init__()
        self.compound_coef = compound_coef
        self.with_centerness = with_centerness
        self.num_anchors = num_anchors
        # Attention here: input num_classes include the background
        self.num_classes = num_classes-1
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]

        self.num_layers = self.box_class_repeats[self.compound_coef]
        self.in_channels = self.fpn_num_filters[self.compound_coef]

        self.regressor = Regressor(self.in_channels,self.num_anchors,self.num_layers,self.with_centerness)
        self.classifier = Classifier(self.in_channels,self.num_anchors,self.num_classes,self.num_layers)

        prior_prob = prior
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.classifier.header.pointwise_conv.conv.bias,bias_value)
        if regression_type == 'POINT':
            torch.nn.init.constant_(self.regressor.header.pointwise_conv.bias,4)


    def forward(self, inputs):
        box_regression, centerness = self.regressor(inputs)
        box_cls = self.classifier(inputs)
        return (box_cls,box_regression, centerness)


class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_layers, with_centerness=True,onnx_export=False):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.num_layers = num_layers
        self.with_centerness = with_centerness

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])

        self.header = SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False)
        if self.with_centerness:
            self.header_centerness = SeparableConvBlock(in_channels, num_anchors * 1, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        if self.with_centerness:
            feats_centerness = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)

            if self.with_centerness:
                feat_centerness = self.header_centerness(feat)
                feats_centerness.append(feat_centerness)

            feat = self.header(feat)

            feats.append(feat)

        return (feats,feats_centerness) if self.with_centerness else (feats,False)


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feats.append(feat)

        return feats