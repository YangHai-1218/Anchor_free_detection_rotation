import torch
from pycocotools import coco,cocoeval
import numpy as np
import json
from utils.boxlist import BoxList,boxlist_ml_nms
from collections import defaultdict

'''
Attention: this test_augemntion pipeline is post prcocess, base on the coco result json format
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
            boxlist = boxlist_ml_nms(boxlist,nms_threshold)

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
    '''
    def __init__(self,weight,object_size_threshold,nms_threshold):
        '''
        weight: list , the first on is for small object weight, the secone one is for larget object weight
        '''
        self.small_object_threshold = object_size_threshold[0]
        self.large_object_threshold = object_size_threshold[1]
        self.weight_for_small_object = weight[0]
        self.weight_for_larget_object = weight[1]
        self.weights = np.array([[self.weight_for_small_object,1,1],
                                [1,1,self.weight_for_larget_object]
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
            boxlist = boxlist_ml_nms(boxlist, self.nms_threshold)

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














