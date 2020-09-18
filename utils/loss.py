import os

import torch
from torch import nn
from utils.boxlist import cat_boxlist, boxlist_iou
from utils.assigner import Assigner
INF = 100000000

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target,weights=None):
        '''
        Args:
            out: tensor shape (N，class_num), no sigmoid
            target: tensor shape (N) the label for every anchor
        '''
        # TODO mixup loss
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p ** gamma * torch.log(1 - p)

        # print(term1.sum(), term2.sum())

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        loss = loss.sum(dim=1)*weights


        return loss.sum()

class GIoULoss(nn.Module):
    def __init__(self, coder):
        super(GIoULoss, self).__init__()
        self.box_coder = coder

    def forward(self, pred, target, anchor, weight):
        '''
        Args:
            pred tensor shape (N,4) N is the number of positive sample, localization branch output,
            target tensor shape (N,4) after coder.encode(gt_bbox,anchors)
            anchor tensor shape (N.4)
            weight tensor shape (N) if mixup , then the weight might <1. No mixup , weight = 1
        '''
        pred_with_angel = torch.cat([pred.view(-1, 4), pred.new_zeros((pred.shape[0], 1))], dim=-1)
        pred_boxes = self.box_coder.decode(pred_with_angel.view(-1, 5), anchor.view(-1, 5))
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2]/2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3]/2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]/2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3]/2

        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]

        target_with_angle = torch.cat([target.view(-1, 4), target.new_zeros((target.shape[0], 1))], dim=-1)

        gt_boxes = self.box_coder.decode(target_with_angle.view(-1, 5), anchor.view(-1, 5))
        target_x1 = gt_boxes[:, 0] - gt_boxes[:, 2]/2
        target_y1 = gt_boxes[:, 1] - gt_boxes[:, 3]/2
        target_x2 = gt_boxes[:, 0] + gt_boxes[:, 2]/2
        target_y2 = gt_boxes[:, 1] + gt_boxes[:, 3]/2
        target_area = gt_boxes[:, 2] * gt_boxes[:, 3]

        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size()).to(pred)
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])

        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-7

        area_union = pred_area + target_area - area_intersect + 1e-7
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing

        losses = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()

class SmoothL1loss_with_weight(nn.Module):
    def __init__(self):
        super(SmoothL1loss_with_weight, self).__init__()
    def forward(self, pred, targets, weights):
        assert pred.shape[0] == targets.shape[0] == weights.shape[0]
        loss = nn.SmoothL1Loss(reduction='none')(pred, targets)
        loss = loss.sum(dim=-1) * weights
        loss = loss.sum()
        return loss



def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer

def concat_box_prediction_layers(box_cls, box_regression):
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(
        box_cls, box_regression
    ):
        N, AxC, H, W = box_cls_per_level.shape
        Ax4 = box_regression_per_level.shape[1]
        A = Ax4 // 4
        C = AxC // A
        box_cls_per_level = permute_and_flatten(
            box_cls_per_level, N, A, C, H, W
        )
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W
        )
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).reshape(-1, C)
    box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


