import torch
from torch import nn
from .Atss_head import ATSSLoss
from utils import boxlist_iou,BoxList
from utils import (
    GIoULoss,
    SigmoidFocalLoss,
    SmoothL1loss_with_weight,
    concat_box_prediction_layers,
    get_num_gpus,
    reduce_sum,
    cat_boxlist,
)

class ATSSLoss_IOU(ATSSLoss):
    def __init__(self, gamma, alpha, fg_iou_threshold, bg_iou_threshold, positive_type,
                 reg_loss_weight, angle_loss_weight, cls_loss_weight,
                 top_k, box_coder):
        super(ATSSLoss_IOU, self).__init__(gamma, alpha, fg_iou_threshold, bg_iou_threshold, positive_type,
                 reg_loss_weight, angle_loss_weight, cls_loss_weight,
                 top_k, box_coder)
        self.iou_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.reg_loss_func = SmoothL1loss_with_weight()
        self.angle_loss_func = nn.CrossEntropyLoss(reduction='sum')

    # IOU aware
    def compute_iou_targets(self, reg_targets, angle_targets, anchors):
        assert reg_targets.shape[0] == angle_targets.shape[0] == anchors.shape[0]
        reg_targets_with_angel = torch.cat([reg_targets.view(-1,4), angle_targets.view(-1, 1)],dim=-1)
        gts = self.box_coder.decode(reg_targets_with_angel, anchors)
        gts = BoxList(gts, image_size=(1024, 1024), mode='xywha_d')
        anchors_box = BoxList(anchors, image_size=(1024,1024), mode='xywha_d')
        ious = boxlist_iou(gts, anchors_box)
        index = torch.linspace(0, anchors.shape[0]-1, anchors.shape[0]).to(torch.long).view(1, -1).to(ious.device)
        ious = ious.gather(0, index).view(-1)
        return ious

    def __call__(self, box_cls, box_regression, iou, angle, targets, anchors):
        '''
        box_cls: list(tensor) tensor shape (N,class_num,H,W) classification branch output for every feature level ,
                        N is the batchsize,
        box_regression : list(tensor) tensor shape (N,4,H,W) localization branch output for every feature level
        iou: list(tensor) tensor shape (N,1.H,W) iou branch output for every feature level
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
        iou_flatten = [ct.permute(0, 2, 3, 1).reshape(N, -1, 1) for ct in iou]
        iou_flatten = torch.cat(iou_flatten, dim=1).reshape(-1)
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

        cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int(),
                                      weights_label_flatten) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            anchors_flatten = anchors_flatten[pos_inds]

            # prepare positive sample matched gt
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            angel_targets_flatten = angel_targets_flatten[pos_inds]
            iou_targets = self.compute_iou_targets(reg_targets_flatten, angel_targets_flatten, anchors_flatten)
            weights_label_flatten = weights_label_flatten[pos_inds]

            # prepare positive sample prediction
            box_regression_flatten = box_regression_flatten[pos_inds]
            iou_flatten = iou_flatten[pos_inds]
            angle_flatten = angle_flatten[pos_inds]

            sum_iou_targets_avg_per_gpu= reduce_sum(iou_targets.sum()).item() / float(num_gpus)

            # attention here
            reg_loss = self.reg_loss_func(box_regression_flatten, reg_targets_flatten,
                                          weights_label_flatten*iou_targets) \
                       / sum_iou_targets_avg_per_gpu
            # reg_loss = self.reg_loss_func(box_regression_flatten, reg_targets_flatten, anchors_flatten,
            #                          weight=iou_targets*weights_label_flatten) \
            #            /sum_iou_targets_avg_per_gpu

            iou_loss = self.iou_loss_func(iou_flatten, iou_targets) / num_pos_avg_per_gpu

            angle_loss = self.angle_loss_func(angle_flatten, angel_targets_flatten.to(torch.long)) / num_pos_avg_per_gpu
        else:
            reg_loss = torch.tensor([0]).to(torch.float32)
            iou_loss = reg_loss * 0
            angle_loss = reg_loss * 0

        return cls_loss * self.cls_loss_weight, reg_loss * self.reg_loss_weight, iou_loss, angle_loss * self.angle_loss_weight