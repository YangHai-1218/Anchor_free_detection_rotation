import torch
from torch import nn
from Head import EfficientDetHead, ATSSLoss, ATSSHead,ATSSLoss_IOU
from FPN.FPN import FPN, FPNTopP6P7
from utils import AnchorGenerator, BoxCoder, ATSSPostProcessor
from backbone import EfficientnetWithBiFPN


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale





def make_anchor_generator_atss(anchor_sizes, anchor_strides, anchor_ratios):
    aspect_ratios = anchor_ratios
    straddle_thresh = 0
    octave = 2.0
    scales_per_octave = 1

    assert len(anchor_strides) == len(anchor_sizes), "Only support FPN now"
    new_anchor_sizes = []
    for size in anchor_sizes:
        per_layer_anchor_sizes = []
        for scale_per_octave in range(scales_per_octave):
            octave_scale = octave ** (scale_per_octave / float(scales_per_octave))
            per_layer_anchor_sizes.append(octave_scale * size)
        new_anchor_sizes.append(tuple(per_layer_anchor_sizes))

    anchor_generator = AnchorGenerator(
        tuple(new_anchor_sizes), aspect_ratios, anchor_strides, straddle_thresh
    )
    return anchor_generator





class ATSS(nn.Module):
    def __init__(self, config, backbone):
        super(ATSS, self).__init__()

        self.backbone = backbone
        fpn_top = FPNTopP6P7(
            config.feat_channels[-1], config.out_channel, use_p5=config.use_p5
            )
        self.fpn = FPN(config.feat_channels, config.out_channel, fpn_top)

        self.head = ATSSHead(
            config.out_channel, config.n_class, config.n_conv, config.prior, config.regression_type
            )

        box_coder = BoxCoder(config.regression_type, config.anchor_sizes, config.anchor_strides)
        # self.loss_evaluator = ATSSLoss(
        #     config.gamma, config.alpha, config.fg_iou_threshold, config.bg_iou_threshold,
        #     config.positive_type, config.reg_loss_weight, config.cls_loss_weight,
        #     config.top_k, box_coder
        #     )
        self.loss_evaluator = ATSSLoss_IOU(
            config.gamma, config.alpha, config.fg_iou_threshold, config.bg_iou_threshold,
            config.positive_type, config.reg_loss_weight, config.angle_loss_weight, config.cls_loss_weight,
            config.top_k, box_coder,
        )
        self.box_selector_test = ATSSPostProcessor(
            config.inference_th, config.pre_nms_top_n, 
            config.nms_threshold, config.detections_per_img, 
            config.min_size, config.n_class, box_coder,
            config.voting_enable, config.multi_scale_test,
            config.voting_threshold,
            )
        self.anchor_generator = make_anchor_generator_atss(
            config.anchor_sizes, config.anchor_strides, config.anchor_ratios
            )

    def forward(self, images, targets=None):

        features = self.backbone(images.tensors)
        # print('backbone extracted')
        # for feature in features:
        #     print(feature.shape)
        features = self.fpn(features)
        # print('fpn extracted')
        # for feature in features:
        #     print(feature.shape)
        box_cls, box_regression, quality, angle = self.head(features)
        anchors = self.anchor_generator(images, features)
 
        if self.training:
            return self._forward_train(box_cls, box_regression, quality, angle, targets, anchors)
        else:
            return self._forward_test(box_cls, box_regression, quality, angle, anchors)

    def _forward_train(self, box_cls, box_regression, quality, angle, targets, anchors):
        loss_box_cls, loss_box_reg, loss_quality, loss_angle = self.loss_evaluator(
            box_cls, box_regression, quality, angle, targets, anchors
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_quality": loss_quality,
            "loss_angle": loss_angle,
        }
        return None, losses

    def _forward_test(self, box_cls, box_regression, quality, angle, anchors):
        boxes = self.box_selector_test(box_cls, box_regression, quality, angle, anchors)
        return boxes, {}


class Efficientnet_Bifpn_ATSS(nn.Module):
    def __init__(self, config, compound_coef=0, load_backboe_weight=False, weight_path=None):
        super(Efficientnet_Bifpn_ATSS, self).__init__()
        self.backbone = EfficientnetWithBiFPN(compound_coef=compound_coef, load_total_weight=load_backboe_weight,
                                                    total_weight_path=weight_path)
        # self.head = ATSSHead(
        #     self.backbone.fpn_num_filters[compound_coef], config.n_class,
        #     config.n_conv, config.prior, config.regression_type
        # )
        self.head = EfficientDetHead(compound_coef=compound_coef, prior=config.prior, num_anchors=1,
                                     regression_type=config.regression_type,
                                     num_classes=config.n_class, with_centerness=True)
        box_coder = BoxCoder(config.regression_type, config.anchor_sizes, config.anchor_strides)
        # self.loss_evaluator = ATSSLoss(
        #     config.gamma, config.alpha, config.fg_iou_threshold, config.bg_iou_threshold,
        #     config.positive_type, config.reg_loss_weight, config.angle_loss_weight, config.top_k, box_coder
        # )
        self.loss_evaluator = ATSSLoss_IOU(
                config.gamma, config.alpha, config.fg_iou_threshold, config.bg_iou_threshold,
                config.positive_type, config.reg_loss_weight, config.angle_loss_weight, config.top_k, box_coder
            )
        self.box_selector_test = ATSSPostProcessor(
            config.inference_th, config.pre_nms_top_n,
            config.nms_threshold, config.detections_per_img,
            config.min_size, config.n_class, box_coder,
            config.voting_enable,config.multi_scale_test,
            config.voting_threshold,
        )
        self.anchor_generator = make_anchor_generator_atss(
            config.anchor_sizes, config.anchor_strides, config.anchor_ratios
        )

    def forward(self, images, targets=None,val_withloss=False):


        features = self.backbone(images.tensors)

        box_cls, box_regression, quality, angle = self.head(features)
        anchors = self.anchor_generator(images, features)

        if self.training:
            return self._forward_train(box_cls, box_regression, quality, angle, targets, anchors)
        else:
            if val_withloss:
                _, losses = self._forward_train(box_cls, box_regression, quality, angle, targets, anchors)

            boxes, _ = self._forward_test(box_cls, box_regression, quality, angle, anchors)
            if val_withloss:
                return boxes, losses
            else:
                return boxes, None

    def _forward_train(self, box_cls, box_regression, quality, angle, targets, anchors):
        loss_box_cls, loss_box_reg, loss_quality, loss_angle = self.loss_evaluator(
            box_cls, box_regression, quality, angle, targets, anchors
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_quality": loss_quality,
            "loss_angle": loss_angle,
        }
        return None, losses

    def _forward_test(self, box_cls, box_regression, quality, angle, anchors):
        boxes = self.box_selector_test(box_cls, box_regression, quality, angle, anchors)
        return boxes, {}


def test():
    make_anchor_generator_atss()