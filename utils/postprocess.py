import torch
from torch import nn

from utils.boxlist import BoxList, boxlist_ml_nms, remove_small_boxes, cat_boxlist
from loss import permute_and_flatten

class ATSSPostProcessor(nn.Module):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        box_coder,
        bbox_aug_enabled=False,
        bbox_aug_vote=False
    ):
        super(ATSSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.box_coder = box_coder
        self.bbox_aug_vote = bbox_aug_vote

    def forward_for_single_feature_map(self, box_cls, box_regression, centerness, anchors):
        N, _, H, W = box_cls.shape
        A = box_regression.size(1) // 4  # num_anchor
        C = box_cls.size(1) // A    # class_num

        # put in the same format as anchors
        # box_cls (N,anchors_feature_map,class_num)
        box_cls = permute_and_flatten(box_cls, N, A, C, H, W)
        box_cls = box_cls.sigmoid()

        # box_regression (N,anchor_feature_map,4)
        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W)
        box_regression = box_regression.reshape(N, -1, 4)

        # filter some anchors using nms_thresh
        # candidate_inds (N,anchors_feature_map,class_num)
        candidate_inds = box_cls > self.pre_nms_thresh

        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        # pre_nms_top_n (N)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)


        centerness = permute_and_flatten(centerness, N, A, 1, H, W)
        centerness = centerness.reshape(N, -1).sigmoid()

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for per_box_cls, per_box_regression, per_pre_nms_top_n, per_candidate_inds, per_anchors \
                in zip(box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            # per_box_cls: (anchors_feature_map,class_num)
            # per_box_regression: (anchors_feature_map,4)
            # per_pre_nms_top_n: shape=(1)
            # per_candidate_inds: (anchors_feature_map,class_num) True or False,
            # True means the prediction probability > inference_threshold

            # per_box_cls (n) n is the number of elements > inference_threshold in per_box_cls, the element is the probability
            per_box_cls = per_box_cls[per_candidate_inds]

            #
            per_box_cls, top_k_indices = per_box_cls.topk(per_pre_nms_top_n, sorted=False)

            # per_candidate_inds.nonzero()   (n,2) n is equal to per_box_cls.shape[0]
            per_candidate_nonzeros = per_candidate_inds.nonzero()[top_k_indices, :]

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1

            detections = self.box_coder.decode(
                per_box_regression[per_box_loc, :].view(-1, 4),
                per_anchors.bbox[per_box_loc, :].view(-1, 4)
            )

            boxlist = BoxList(detections, per_anchors.size, mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, box_cls, box_regression, centerness, anchors):
        '''
        params:
        box_cls: model classifier output for every feature map,
                list [p3_predict, p4_predict,p5_predict,p6_predict,p7_predict]
                pi_predict: tensor (N,num_anchors,H,W)  N is the batchsize
        box_regression: model regressor output for every feature map
                list [p3_predict,p4_predict,p5_predict,p6_predict,p7_predict]
                pi_predict: tensor (N,num_anchors*4,H,W)
        centerness: model centerness branch ourpur for every feature map
                list [p3_predict,p4_predict,p5_predict,p6_predict,p7_predict]
                pi_predict: tensor (N,num_anchors,H,w)
        acnhors: anchor loaction for every image (in the batchsize) and every feature map
                list [ima1_anchor,...imageN_anchors]
                imgi_anchor: list[p3_anchors,p4_anchors,p5_anchors,p6_anchors,p7_anchors]
                pi_anchors: Boslist object
        '''
        sampled_boxes = []
        anchors = list(zip(*anchors))
        for _, (o, b, c, a) in enumerate(zip(box_cls, box_regression, centerness, anchors)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(o, b, c, a)
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not (self.bbox_aug_enabled and not self.bbox_aug_vote):
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

