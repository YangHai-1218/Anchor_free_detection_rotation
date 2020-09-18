import torch
from .boxlist import cat_boxlist,boxlist_iou
import copy
INF = 100000000



class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class Assigner:
    def __init__(self,mode,box_coder,fg_iou_threshold, bg_iou_threshold,top_k,**kwargs):
        '''
        lable assigner
        mode: 'SSC', 'ATSS', 'TOPK' 'IoU'
        '''
        self.mode = mode
        self.box_coder = box_coder
        self.matcher = Matcher(fg_iou_threshold,bg_iou_threshold,True)
        self.args_other = kwargs
        self.top_k = top_k



    def targets_prepare(self,targets):
        targets_ = copy.deepcopy(targets)
        rounded_targets = targets_.xywhad_round()
        rounded_targets.bbox[:, 4] += 90
        return rounded_targets


    def ATSS_assign(self, targets, anchors):
        '''
        Args:
            targets: list(boxlist), gt_bbox for images in batch
            anchos: list(list)  [image_1_anchors,...,image_N_anchors],
                image_i_anchors : [leverl_1_anchor,...,leverl_n_anchor]
                level_i_anchor:boxlist
        '''
        cls_labels = []
        reg_targets = []
        weights = []

        for im_i in range(len(targets)):

            targets_per_im = targets[im_i]
            assert targets_per_im.mode == 'xywha_d'


            xyxyxyxy_bboxes_per_im = targets_per_im.convert('xyxyxyxy').bbox
            rounded_targets_per_im = self.targets_prepare(targets_per_im)
            bboxes_per_im = rounded_targets_per_im.bbox
            labels_per_im = rounded_targets_per_im.get_field("labels")


            try:
                weights_per_im = rounded_targets_per_im.get_field("weights")
            except:
                weights_per_im = rounded_targets_per_im.bbox.new_ones(len(targets_per_im))

            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            num_anchors_per_level = [len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]
            ious = boxlist_iou(anchors_per_im, targets_per_im)

            gt_cx = bboxes_per_im[:, 0]
            gt_cy = bboxes_per_im[:, 1]
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = anchors_per_im.bbox[:, 0]
            anchors_cy_per_im = anchors_per_im.bbox[:, 1]
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # Selecting candidates based on the center distance between anchor box and object
            candidate_idxs = []
            star_idx = 0
            for level, anchors_per_level in enumerate(anchors[im_i]):
                end_idx = star_idx + num_anchors_per_level[level]
                distances_per_level = distances[star_idx:end_idx, :]
                topk = min(self.top_k, num_anchors_per_level[level])
                _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
                candidate_idxs.append(topk_idxs_per_level + star_idx)
                star_idx = end_idx
            candidate_idxs = torch.cat(candidate_idxs, dim=0)

            # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
            candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # Limiting the final positive samples’ center to object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                candidate_idxs[:, ng] += ng * anchor_num


            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            candidate_idxs = candidate_idxs.view(-1)
            l = torch.stack([e_anchors_cx[candidate_idxs].view(-1, num_gt) - xyxyxyxy_bboxes_per_im[:, 0],
                             e_anchors_cx[candidate_idxs].view(-1, num_gt) - xyxyxyxy_bboxes_per_im[:, 2]], dim=-1)
            l= torch.mean(l, dim=-1)
            t = torch.stack([e_anchors_cy[candidate_idxs].view(-1, num_gt) - xyxyxyxy_bboxes_per_im[:, 1],
                             e_anchors_cy[candidate_idxs].view(-1, num_gt) - xyxyxyxy_bboxes_per_im[:, 7]], dim=-1)
            t= torch.mean(t, dim=-1)
            #t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            #r = bboxes_per_im[:, 0] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
            r = torch.stack([xyxyxyxy_bboxes_per_im[:, 4] - e_anchors_cx[candidate_idxs].view(-1, num_gt),
                             xyxyxyxy_bboxes_per_im[:, 6] - e_anchors_cx[candidate_idxs].view(-1, num_gt)], dim=-1)
            r= torch.mean(r, dim=-1)
            #b = bboxes_per_im[:, 1] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
            b = torch.stack([xyxyxyxy_bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt),
                             xyxyxyxy_bboxes_per_im[:, 5] - e_anchors_cy[candidate_idxs].view(-1, num_gt)], dim=-1)
            b= torch.mean(b, dim=-1)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            is_pos = is_pos & is_in_gts

            # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = candidate_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index]
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0

            matched_gts = bboxes_per_im[anchors_to_gt_indexs]


            # for mixup, weight loss
            weight = weights_per_im[anchors_to_gt_indexs]
            weight[anchors_to_gt_values == -INF] = 1

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)

            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)
            weights.append(weight)

        return cls_labels, reg_targets, weights

    def SSC_assign(self,targets,anchors):
        cls_labels = []
        reg_targets = []
        weights = []

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            try:
                weights_per_im = targets_per_im.get_field("weights")
            except:
                weights_per_im = targets_per_im.bbox.new_ones(len(targets_per_im))

            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]


            object_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]
            area_per_im = targets_per_im.area()
            expanded_object_sizes_of_interest = []
            points = []
            for l, anchors_per_level in enumerate(anchors[im_i]):
                anchors_per_level = anchors_per_level.bbox
                anchors_cx_per_level = (anchors_per_level[:, 2] + anchors_per_level[:, 0]) / 2.0
                anchors_cy_per_level = (anchors_per_level[:, 3] + anchors_per_level[:, 1]) / 2.0
                points_per_level = torch.stack((anchors_cx_per_level, anchors_cy_per_level), dim=1)
                points.append(points_per_level)
                object_sizes_of_interest_per_level = \
                    points_per_level.new_tensor(object_sizes_of_interest[l])
                expanded_object_sizes_of_interest.append(
                    object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
                )
            expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
            points = torch.cat(points, dim=0)


            # attention: broadcasting
            xs, ys = points[:, 0], points[:, 1]
            l = xs[:, None] - bboxes_per_im[:, 0][None]
            t = ys[:, None] - bboxes_per_im[:, 1][None]
            r = bboxes_per_im[:, 2][None] - xs[:, None]
            b = bboxes_per_im[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0.01

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # attention here
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= expanded_object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= expanded_object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area_per_im[None].repeat(len(points), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            cls_labels_per_im = labels_per_im[locations_to_gt_inds]
            cls_labels_per_im[locations_to_min_area == INF] = 0
            matched_gts = bboxes_per_im[locations_to_gt_inds]
            # for mixup, weight loss
            weight = weights_per_im[locations_to_gt_inds]
            weight[locations_to_min_area == INF] = 1

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

            weights.append(weight)

        return cls_labels, reg_targets, weights


    def TopK_assign(self,targets,anchors,):
        cls_labels = []
        reg_targets = []
        weights = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            try:
                weights_per_im = targets_per_im.get_field("weights")
            except:
                weights_per_im = targets_per_im.bbox.new_ones(len(targets_per_im))

            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
            distances = distances / distances.max() / 1000
            ious = boxlist_iou(anchors_per_im, targets_per_im)

            is_pos = ious * False
            for ng in range(num_gt):
                _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(self.top_k, dim=0)
                l = anchors_cx_per_im[topk_idxs] - bboxes_per_im[ng, 0]
                t = anchors_cy_per_im[topk_idxs] - bboxes_per_im[ng, 1]
                r = bboxes_per_im[ng, 2] - anchors_cx_per_im[topk_idxs]
                b = bboxes_per_im[ng, 3] - anchors_cy_per_im[topk_idxs]
                is_in_gt = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
                is_pos[topk_idxs[is_in_gt == 1], ng] = True

            ious[is_pos == 0] = -INF
            anchors_to_gt_values, anchors_to_gt_indexs = ious.max(dim=1)

            cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            cls_labels_per_im[anchors_to_gt_values == -INF] = 0
            matched_gts = bboxes_per_im[anchors_to_gt_indexs]
            # for mixup, weight loss
            weight = weights_per_im[anchors_to_gt_indexs]
            weight[anchors_to_gt_values == -INF] = 1

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

            weights.append(weight)

        return cls_labels, reg_targets, weights


    def Iou_assign(self,targets,anchors):
        cls_labels = []
        reg_targets = []
        weights = []
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes_per_im = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")

            try:
                weights_per_im = targets_per_im.get_field("weights")
            except:
                weights_per_im = targets_per_im.bbox.new_ones(len(targets_per_im))

            anchors_per_im = cat_boxlist(anchors[im_i])
            num_gt = bboxes_per_im.shape[0]

            match_quality_matrix = boxlist_iou(targets_per_im, anchors_per_im)
            matched_idxs = self.matcher(match_quality_matrix)
            targets_per_im = targets_per_im.copy_with_fields(['labels'])
            matched_targets = targets_per_im[matched_idxs.clamp(min=0)]

            cls_labels_per_im = matched_targets.get_field("labels")
            cls_labels_per_im = cls_labels_per_im.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            cls_labels_per_im[bg_indices] = 0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            cls_labels_per_im[inds_to_discard] = -1

            matched_gts = matched_targets.bbox

            # for mixup, weight loss
            weight = weights_per_im[matched_idxs.clamp(min=0)]
            weight[bg_indices] = 1

            # Limiting positive samples’ center to object
            # in order to filter out poor positives and use the centerness branch
            pos_idxs = torch.nonzero(cls_labels_per_im > 0).squeeze(1)
            pos_anchors_cx = (anchors_per_im.bbox[pos_idxs, 2] + anchors_per_im.bbox[pos_idxs, 0]) / 2.0
            pos_anchors_cy = (anchors_per_im.bbox[pos_idxs, 3] + anchors_per_im.bbox[pos_idxs, 1]) / 2.0
            l = pos_anchors_cx - matched_gts[pos_idxs, 0]
            t = pos_anchors_cy - matched_gts[pos_idxs, 1]
            r = matched_gts[pos_idxs, 2] - pos_anchors_cx
            b = matched_gts[pos_idxs, 3] - pos_anchors_cy
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
            cls_labels_per_im[pos_idxs[is_in_gts == 0]] = -1

            reg_targets_per_im = self.box_coder.encode(matched_gts, anchors_per_im.bbox)
            cls_labels.append(cls_labels_per_im)
            reg_targets.append(reg_targets_per_im)

            weights.append(weight)

        return cls_labels, reg_targets, weights,


    def __call__(self, targets, anchors):
        '''
        targets: list(boxlist) , boxlist object, ground_truth object for every image
        anchos: list(list)  [image_1_anchors,...,image_N_anchors],
                image_i_anchors : [leverl_1_anchor,...,leverl_n_anchor]
                level_i_anchor:boxlist ,
        output:
            cls_labels: list(tensor), tensor shape (n,), n is the number of anchors for every image
                        [image_1_anchor_labels,...,image_N_anchor_labels] ,
                        label value is in interval [-1,0,1,....,class_num]
                        0 means the background(negative) sample
                        -1 means we will ignore the anchor when calculating localization loss
                            and classification loss and other loss
                        [0,....,class_num]  means the positive sample
            reg_targets: list(tensor) tensor shaple (n,4)
            weight_labels: list(tensor) tensor shape (n,) , weight_value is in interval (0,1] ,
                            if the anchor is a positive sample, when calculating loss(classification and locaization),
                            the anchor loss =  the original anchor loss * weight
        '''

        if self.mode == 'SSC':
            cls_labels, reg_targets, weights = self.Iou_assign(targets,anchors)
        elif self.mode == 'ATSS':
            cls_labels, reg_targets, weights = self.ATSS_assign(targets,anchors)
        elif self.mode == 'TOPK':
            cls_labels, reg_targets, weights = self.TopK_assign(targets, anchors)
        elif self.mode == 'IOU':
            cls_labels, reg_targets, weights = self.Iou_assign(targets, anchors)
        else:
            raise  NotImplementedError

        return cls_labels, reg_targets, weights
