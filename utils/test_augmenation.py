import torch
from pycocotools import coco,cocoeval
import numpy as np
import json
from utils import BoxList,boxlist_ml_rnms,cat_boxlist,ImageList
from collections import defaultdict
import os
import cv2
from utils.transform import ToTensor,RandomHorizontalFlip,Resize_letterbox_square
from evaluate import map_to_origin_image
from .dataset import ImageList
'''
Attention: this test_augmenation pipeline is post process, base on the coco result json format
'''


class Ensemble:
    '''
    https://arxiv.org/abs/2008.01365
    '''
    def __init__(self,gt_path,det_paths,theta0=0.5):
        self.coco_gt = coco.COCO(gt_path)
        self.coco_dts = [self.coco_gt.loadRes(det_path)  for det_path in det_paths]
        self.coco_evals = [cocoeval.COCOeval(self.coco_gt,coco_dt,iouType='bbox')for coco_dt in self.coco_dts]
        self.num_candidates = len(det_paths)
        self.theta0 = theta0

    def coco_eval(self):
        self.coco_aps = []
        for coco_eval in self.coco_evals:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            coco_ap = coco_eval.stats[0]
            self.coco_aps.append(coco_ap)
        self.coco_aps = np.array(self.coco_aps)

    def get_weights(self):
        rank = np.argsort(-self.coco_aps)
        weights = {rank[i-1]:self.theta0 + (self.num_candidates-i) * (1-self.theta0)/(self.num_candidates-1)
                   for i in range(1,self.num_candidates+1)}
        self.weights = weights
        return weights

    @classmethod
    def ensemble(cls,det_paths,weights,nms_threshold):

        images_bbox = defaultdict(dict)
        for i,det_path in enumerate(det_paths):
            det_result = json.load(det_path)

            for result in det_result:
                image_id = result['image_id']
                bbox = result['bbox']
                label = result['category_id']
                score = result['score']

                score *= weights[i]

                if image_id not in images_bbox:
                    images_bbox[image_id]['bbox'] = [bbox]
                    images_bbox[image_id]['labels'] = label
                    images_bbox[image_id]['scores'] = score,
                else:
                    images_bbox[image_id]['bbox'].append(bbox)
                    images_bbox[image_id]['labels'].append(label)
                    images_bbox[image_id]['scores'].append(score)


        nms_result = []
        for image_bbox in images_bbox:
            image_id = image_bbox
            # the image_size may be a bug
            boxlist = BoxList(images_bbox[image_id]['bbox'],image_size=(1080,1920),mode='xywh').convert('xyxy')
            boxlist.add_field('scores',torch.tensor(images_bbox[image_id]['scores']))
            boxlist.add_field('labels',torch.tensor(images_bbox[image_id]['labels']))
            # nms
            boxlist = boxlist_ml_rnms(boxlist,nms_threshold)

            boxlist = boxlist.convert('xywh')
            boxes = boxlist.bbox.tolist()
            labels = boxlist.get_field('labels').tolist()
            scores = boxlist.get_field('scores').tolist()
            nms_result.extend(
                [
                    {
                        'image_id': image_id,
                        'category_id': labels[k],
                        'bbox': boxes[k],
                        'score': scores[k],
                    }
                    for k in range(len(boxlist))
                ]
            )


        with open('ensbmble.json') as f:
            json.dump(nms_result,f)




class Multi_Scale_Augment:
    '''
    scale-aware for multi scale testing
    https://arxiv.org/abs/2008.01365
    this method can perform multi_scale_augment offline by using the coco format result json file
    '''
    def __init__(self, weight, object_size_threshold,nms_threshold):
        '''
        weight: list , the first on is for small object weight, the secone one is for larget object weight
        '''
        self.small_object_threshold = object_size_threshold[0]
        self.large_object_threshold = object_size_threshold[1]
        self.weight_for_small_object = weight[0]
        self.weight_for_larget_object = weight[1]
        self.weights = np.array([[self.weight_for_small_object, 1, 1],
                                [1, 1, self.weight_for_larget_object]
                                ])
        self.nms_threshold = nms_threshold



    def ensemble(self,det_paths):
        '''
        det_paths: list , the first one is the small scale, the last one is the larget scale
        '''
        images_bbox = defaultdict(dict)

        for i, det_path in enumerate(det_paths):
            det_result = json.load(det_path)

            for result in det_result:
                image_id = result['image_id']
                bbox = result['bbox']
                label = result['category_id']
                score = result['score']

                if bbox[2]*bbox[3] < self.small_object_threshold:
                    score *= self.weights[i][0]
                elif bbox[2]*bbox[3] > self.large_object_threshold:
                    score *= self.weights[i][2]
                else:
                    score *= self.weights[i][1]


                if image_id not in images_bbox:
                    images_bbox[image_id]['bbox'] = [bbox]
                    images_bbox[image_id]['labels'] = label
                    images_bbox[image_id]['scores'] = score,
                else:
                    images_bbox[image_id]['bbox'].append(bbox)
                    images_bbox[image_id]['labels'].append(label)
                    images_bbox[image_id]['scores'].append(score)


        nms_result = []
        for image_bbox in images_bbox:
            image_id = image_bbox
            # the image_size may be a bug
            boxlist = BoxList(images_bbox[image_id]['bbox'], image_size=(1080, 1920), mode='xywh').convert('xyxy')
            boxlist.add_field('scores', torch.tensor(images_bbox[image_id]['scores']))
            boxlist.add_field('labels', torch.tensor(images_bbox[image_id]['labels']))
            # nms
            boxlist = boxlist_ml_rnms(boxlist, self.nms_threshold)

            boxlist = boxlist.convert('xywh')
            boxes = boxlist.bbox.tolist()
            labels = boxlist.get_field('labels').tolist()
            scores = boxlist.get_field('scores').tolist()
            nms_result.extend(
                [
                    {
                        'image_id': image_id,
                        'category_id': labels[k],
                        'bbox': boxes[k],
                        'score': scores[k],
                    }
                    for k in range(len(boxlist))
                ]
            )

        with open('ensbmble.json') as f:
            json.dump(nms_result, f)



class Multi_Scale_Test:
    def __init__(self,

                 dataset,
                 scale,
                 weight,
                 object_size_threshold,
                 nms_threshold,
                 fpn_post_nms_top_n,
                 device,
                 flip_enable=False,
                 bbox_aug_vote=False,
                 bbox_voting_threshold=0,
                 normalize=None):
        """
        dataset : dataset object
        scale : list[tuple] , (width,height), inference on every fixed scale
        weight: list[list] , [small_object_weight,medium_object_weight,larget_object_weight], for every scale,
                the correspoding weight for different size objects
        object_size_threshold: list [small_object_max_size, larget_object_min_size]
        nms_threshold: the threshold for nms_threshold
        fpn_post_nms_top_n : after nms , keep n bbox with highest bbox
        flip_enable: for every scale , also perform inference on fliped image
        bbox_aug_vote: enable bbox voting
        bbox_voting_threshold: the threshold for voting , use the bbox weith IOU>this threshold to refine bbox
        """
        self.scales = scale
        assert len(scale) == len(weight) , "scale num == weight num"
        self.weights = np.array(weight)
        self.small_object_threshold = object_size_threshold[0]
        self.large_object_threshold = object_size_threshold[1]
        self.dataset = dataset
        self.flip = flip_enable
        self.nms_threshold = nms_threshold
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.bbox_aug_vote = bbox_aug_vote
        self.bbox_voting_threshold = bbox_voting_threshold
        self.device = device
        self.normalize = normalize


    def weight_score(self, weight, preds):
        preds_new = []
        for pred in preds:
            if pred is None:
                preds_new.append(None)
                continue
            scores = pred.get_field("scores")
            area = pred.area()

            # weight score
            small_object_index = area < self.small_object_threshold
            scores[small_object_index] *= weight[0]
            medium_object_index = (self.small_object_threshold <= area)& (area < self.large_object_threshold)
            scores[medium_object_index] *= weight[1]
            large_object_inedx = area >= self.large_object_threshold
            scores[large_object_inedx] *= weight[2]
            pred.add_field("scores", scores)
            preds_new.append(pred)
        return preds_new

    def __call__(self, model, images, indexs):
        # TODO VOTING

        images_resized = {i: [] for i, _ in enumerate(self.scales)}

        images_resized_flipped = {i: [] for i, _ in enumerate(self.scales)}


        for i, scale in enumerate(self.scales):
            assert scale[0] == scale[1], "only support square letterbox resize"
            for index in indexs:
                image_meta = self.dataset.get_image_meta(index)
                image_path = image_meta['file_name']
                image_path = os.path.join(self.dataset.root, image_path)
                image_origin = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


                image_resized, _ = Resize_letterbox_square(size=scale[0])(image_origin, None)
                image_resized_tensor, _ = ToTensor()(image_resized, None)
                image_resized_tensor, _ = self.normalize(image_resized_tensor, None)
                images_resized[i].append(image_resized_tensor)

                image_resized_flipped, _ = RandomHorizontalFlip(p=1)(image_resized, None)
                image_resized_fliped_tensor, _ = ToTensor()(image_resized_flipped, None)
                image_resized_fliped_tensor, _ = self.normalize(image_resized_fliped_tensor, None)
                images_resized_flipped[i].append(image_resized_fliped_tensor)



        for i in range(len(self.scales)):
            resized_batch = images_resized[i][0].new_zeros((len(indexs),)+images_resized[i][0].shape)
            for img, batch_img in zip(images_resized[i], resized_batch):
                batch_img.copy_(img)

            sizes = [img.shape[-2:] for img in resized_batch]
            images_resized[i] = ImageList(resized_batch,
                                          sizes=sizes)

            resized_flipped_batch = images_resized_flipped[i][0].new_zeros((len(indexs),)+images_resized_flipped[i][0].shape)
            for img, batch_img in zip(images_resized_flipped[i], resized_flipped_batch):
                batch_img.copy_(img)

            sizes = [img.shape[-2:] for img in resized_batch]
            images_resized_flipped[i] = ImageList(resized_flipped_batch,
                                                  sizes=sizes)

        boxlists = []
        for i in range(len(self.scales)):
            resized_preds, _ = model(images_resized[i].to(self.device))
            resized_preds = self.weight_score(self.weights[i], resized_preds)

            resized_flipped_preds, _ = model(images_resized_flipped[i].to(self.device))
            resized_flipped_preds = self.weight_score(self.weights[i], resized_flipped_preds)
            resized_flipped_preds = [pred.transpose(0) if (pred is not None) else None for pred in resized_flipped_preds]

            preds = list(zip(*[resized_preds, resized_flipped_preds]))
            preds = [cat_boxlist(pred) for pred in preds]
            nms_preds = [boxlist_ml_rnms(pred, self.nms_threshold) for pred in preds]
            nms_preds = [nms_pred.convert('xyxyxyxy') for nms_pred in nms_preds]
            maped_preds = [map_to_origin_image(
                self.dataset.get_image_meta(indexs[i]), pred) for i, pred in enumerate(nms_preds)]

            boxlists.append(maped_preds)
        total_boxlists = list(zip(*boxlists))
        total_boxlists = [cat_boxlist(boxlist) for boxlist in total_boxlists]

        results = []
        for boxlist in total_boxlists:
            nms_preds = boxlist_ml_rnms(boxlist, self.nms_threshold)
            number_of_detections = len(nms_preds)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = nms_preds.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                nms_preds = nms_preds[keep]
            results.append(nms_preds)
        return results