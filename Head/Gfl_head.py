import torch
from torch import nn
from utils.base_conv import Scale
import torch.nn.functional as F
from loss import GIoULoss,SigmoidFocalLoss,concat_box_prediction_layers,get_num_gpus,reduce_sum,permute_and_flatten
from utils.assigner import Assigner
from utils.boxlist import cat_boxlist,boxlist_iou,BoxList


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x



class Gfl_head(nn.Module):
    """
    https://arxiv.org/abs/2006.04388
    Args:
        in_channels(int): Number of channels in the input feature map.
        n_class(int):  number of categories excluding the background category.
        n_conv(int):  Number of conv layers in cls and reg tower.
    """
    def __init__(self,in_channels, n_class, n_conv, num_anchors):

        super(Gfl_head,self).__init__()

        self.in_channels = in_channels
        self.n_class = n_class - 1
        self.n_conv = n_conv
        self.num_anchors = num_anchors


        cls_tower = []
        bbox_tower = []
        for i in range(self.n_conv):
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
        self.gfl_cls = nn.Conv2d(
            in_channels, self.num_anchors* self.n_class, kernel_size=3, stride=1,
            padding=1
        )
        self.gfl_reg = nn.Conv2d(
            in_channels,self.num_anchors*4,kernel_size=3,stride=1,
            padding=1
        )
        # default: use five levels
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.gfl_cls, self.gfl_reg,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)


    def forawrd(self,x):
        '''
        Args:
            x list(tensor) tensor shape (N,in_channels,H,W)  Features from the upstream network
        output:
            cls_score list(tensor): tensor shape (N,class_num,H,W)
            bbox_reg list(tensor) : tensor shape (N,4*(reg_max+1),H,W) N is the batch size
        '''

        cls_score = []
        bbox_reg = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            cls_score.append(self.gfl_cls(cls_tower))
            bbox_pred = self.scales[l](self.gfl_reg(box_tower))
            bbox_reg.append(bbox_pred)

        return cls_score,bbox_reg


class Gfl_Loss:
    def _init__(self,fg_iou_threshold,bg_iou_threshold, positive_type, reg_loss_weight, top_k, box_coder,class_num):
        self.bbox_loss = GIoULoss()
        self.assigner = Assigner(positive_type, box_coder, fg_iou_threshold, bg_iou_threshold, {'top_k': top_k})
        self.reg_loss_weight = reg_loss_weight
        self.box_coder = box_coder
        self.class_num = class_num # no background

    def compute_centerness_targets(self, reg_targets, anchors):
        gts = self.box_coder.decode(reg_targets, anchors)
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l = anchors_cx - gts[:, 0]
        t = anchors_cy - gts[:, 1]
        r = gts[:, 2] - anchors_cx
        b = gts[:, 3] - anchors_cy
        left_right = torch.stack([l, r], dim=1)
        top_bottom = torch.stack([t, b], dim=1)
        centerness = torch.sqrt((left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]))
        assert not torch.isnan(centerness).any()
        return centerness

    def label_iou_cal(self,anchors,reg_targets,labels):
        labels_iou = []
        for im_i in range(len(anchors)):

            anchors_per_im = cat_boxlist(anchors[im_i])
            label_iou = reg_targets[im_i].new_zeros( (len(anchors_per_im),))

            matched_gts_per_im = self.box_coder.decode(reg_targets[im_i],anchors_per_im)
            matched_gts_per_im = BoxList(matched_gts_per_im,anchors_per_im.size,mode='xyxy')
            iou = boxlist_iou(anchors_per_im,matched_gts_per_im)
            pos_inds = labels>0
            label_iou[pos_inds] = iou[pos_inds]

            labels_iou.append(label_iou)
        return labels_iou

    def quality_focal_loss(self,pred, target, beta=2.0):
        r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
        Qualified and Distributed Bounding Boxes for Dense Object Detection
        <https://arxiv.org/abs/2006.04388>`_.
        Args:
            pred (torch.Tensor): Predicted joint representation of classification
                and quality (IoU) estimation with shape (N, C), C is the number of
                classes.
            target (tuple([torch.Tensor])): Target category label with shape (N,)
                and target quality label with shape (N,).
            beta (float): The beta parameter for calculating the modulating factor.
                Defaults to 2.0.
        Returns:
            torch.Tensor: Loss tensor with shape (N,).
        """
        assert len(target) == 2, """target for QFL must be a tuple of two elements,
            including category label and quality label, respectively"""
        # label denotes the category id, score denotes the quality score
        label, score = target

        label[label==0] = self.class_num + 1
        label = label - 1

        # negatives are supervised by 0 quality score
        pred_sigmoid = pred.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred.shape)
        loss = F.binary_cross_entropy_with_logits(
            pred, zerolabel, reduction='none') * scale_factor.pow(beta)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = pred.size(1)
        pos = ((label >= 0) & (label < bg_class_ind)).nonzero().squeeze(1)
        pos_label = label[pos].long()
        # positives are supervised by bbox quality (IoU) score
        scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
        loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
            pred[pos, pos_label], score[pos],
            reduction='none') * scale_factor.abs().pow(beta)

        loss = loss.sum()
        return loss

    def __call__(self, box_cls, box_regression, targets, anchors):
        '''
            box_cls: list(tensor) tensor shape (N,class_num,H,W)  N is the batchsize,
                    classification branch output for every feature level ,
            box_regression : list(tensor) tensor shape (N,4,H,W)
                    localization branch output for every feature level
            taregts: list(boxlist) , boxlist object, ground_truth object for every image,
            anchors: list(list)  [image_1_anchors,...,image_N_anchors],
                    image_i_anchors : [leverl_1_anchor,...,leverl_n_anchor]
                    level_i_anchor:boxlist
        '''
        labels, reg_targets, weights_label = self.assigner(targets, anchors)
        N = len(labels) # image_num

        # labels_iou : list(tensor) tensor shape (n,) quality score
        labels_iou = self.label_iou_cal(anchors,reg_targets,labels)

        box_cls_flatten, box_regression_flatten = concat_box_prediction_layers(box_cls, box_regression)
        labels_iou_flatten = torch.cat(labels_iou, dim=0)
        labels_flatten = torch.cat(labels,dim=0)
        reg_targets_flatten = torch.cat(reg_targets, dim=0)
        weights_label_flatten = torch.cat(weights_label, dim=0)
        anchors_flatten = torch.cat([cat_boxlist(anchors_per_image).bbox for anchors_per_image in anchors], dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.quality_focal_loss(box_cls_flatten,[labels,labels_iou],beta=2.0) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            box_regression_flatten = box_regression_flatten[pos_inds]
            reg_targets_flatten = reg_targets_flatten[pos_inds]
            anchors_flatten = anchors_flatten[pos_inds]

            weights_label_flatten = weights_label_flatten[pos_inds]

            centerness_targets = self.compute_centerness_targets(reg_targets_flatten, anchors_flatten)

            sum_centerness_targets_avg_per_gpu = reduce_sum(centerness_targets.sum()).item() / float(num_gpus)



            # attention here
            reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
                                     weight=centerness_targets*weights_label_flatten) / sum_centerness_targets_avg_per_gpu


        else:
            reg_loss = box_regression_flatten.sum()

        return cls_loss,  reg_loss






