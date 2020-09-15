import torch
from torch import nn
import math
from torch.nn import functional as F
from utils import cat_boxlist, Assigner
from utils import GIoULoss,SigmoidFocalLoss,concat_box_prediction_layers,get_num_gpus,reduce_sum

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
        # -90 < angle <= 0, channel = 90
        self.angle = nn.Conv2d(
            in_channels, num_anchors * 90, kernel_size=3, stride=1,
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
        angle = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.regression_type == 'POINT':
                bbox_pred = F.relu(bbox_pred)
            bbox_reg.append(bbox_pred)

            angle.append(self.angle(box_tower))
            centerness.append(self.centerness(box_tower))
        return logits, bbox_reg, centerness, angle


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()

        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


class ATSSLoss(object):
    def __init__(self, gamma, alpha, fg_iou_threshold, bg_iou_threshold, positive_type,
                 reg_loss_weight, angle_loss_weight,
                 top_k, box_coder):
        self.cls_loss_func = SigmoidFocalLoss(gamma, alpha)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.angle_loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.reg_loss_func = GIoULoss(box_coder)
        self.reg_loss_weight = reg_loss_weight
        self.angle_loss_weight = angle_loss_weight
        self.box_coder = box_coder
        self.assigner = Assigner(positive_type,box_coder,fg_iou_threshold,bg_iou_threshold,top_k)


    def compute_centerness_targets(self, reg_targets, anchors):
        reg_targets_with_angel = torch.cat([reg_targets, reg_targets.new_zeros(reg_targets.shape[0],1)],dim=-1)
        gts = self.box_coder.decode(reg_targets_with_angel, anchors)
        anchors_cx = anchors[:, 0]
        anchors_cy = anchors[:, 1]
        gts_left_x = gts[:,0] - gts[:,2]/2
        gts_right_x = gts[:,0] + gts[:,2]/2
        gts_upper_y = gts[:,1] - gts[:,3]/2
        gts_bottom_y = gts[:,1] + gts[:,3]/2
        l = anchors_cx - gts_left_x
        t = anchors_cy - gts_upper_y
        r = gts_right_x - anchors_cx
        b = gts_bottom_y - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def __call__(self, box_cls, box_regression, centerness, angle, targets, anchors):
        '''
        box_cls: list(tensor) tensor shape (N,class_num,H,W) classification branch output for every feature level ,
                        N is the batchsize,
        box_regression : list(tensor) tensor shape (N,4,H,W) localization branch output for every feature level
        centerness: list(tensor) tensor shape (N,1.H,W) centerness branch output for every feature level
        angle: list(tensor) tensor shape (N,90,H,W) angle branch output for every feature level
        taregts: list(boxlist) , boxlist object, ground_truth object for every image,
        anchos: list(list)  [image_1_anchors,...,image_N_anchors],
                image_i_anchors : [leverl_1_anchor,...,leverl_n_anchor]
                level_i_anchor:boxlist
        '''

        labels, reg_targets, weights_label = self.assigner(targets, anchors)

        # prepare prediction
        N = len(labels)
        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(box_cls, box_regression)
        centerness_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in centerness]
        centerness_flatten = torch.cat(centerness_flatten, dim=1).reshape(-1)
        angle_flatten = [an.permute(0, 2, 3, 1).reshape(N, -1, 90) for an in angle]
        angle_flatten = torch.cat(angle_flatten, dim=1).reshape(-1, 90)

        # prepare ground truth
        labels_flatten = torch.cat(labels, dim=0)
        reg_targets_flatten = torch.cat([reg_target[:, :4] for reg_target in reg_targets], dim=0)
        angel_targets_flatten = torch.cat([reg_target[:, 4] for reg_target in reg_targets], dim=0)
        weights_label_flatten = torch.cat(weights_label, dim=0)

        # prepare anchors
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int(),weights_label_flatten) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            anchors_flatten = anchors_flatten[pos_inds]

            # prepare positive sample matched gt
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            angel_targets_flatten = angel_targets_flatten[pos_inds]
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)
            weights_label_flatten = weights_label_flatten[pos_inds]

            # prepare positive sample prediction
            box_regression_flatten = box_regression_flatten[pos_inds]
            centerness_flatten = centerness_flatten[pos_inds]
            angle_flatten = angle_flatten[pos_inds]

            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            # attention here
            reg_loss = self.reg_loss_func(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets*weights_label_flatten) \
                       /sum_centerness_targets_avg_per_gpu

            centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu


            angle_loss = self.angle_loss_func(angle_flatten, angel_targets_flatten.to(torch.long)) / num_pos_avg_per_gpu
        else:
            reg_loss = torch.tensor([0]).to(torch.float32)
            centerness_loss = reg_loss * 0
            angle_loss = reg_loss * 0

        return cls_loss, reg_loss * self.reg_loss_weight, centerness_loss, angle_loss * self.angle_loss_weight



def test():
    from utils import BoxCoder
    coder = BoxCoder(regression_type='bbox',anchor_sizes=[64, 128, 256, 512, 1024],
                     anchor_strides=[8, 16, 32, 64, 128])
    loss_obj = ATSSLoss(gamma=2.0, alpha=0.25, fg_iou_threshold=0.5,bg_iou_threshold=0.4,positive_type='ATSS',
                        reg_loss_weight=2.0, top_k=9, box_coder=coder)
    batchsize = 1
    H = 12
    W = 12
    class_num = 2