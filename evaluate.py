import json
import tempfile
from collections import OrderedDict
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils import BoxList
#from utils.pycocotools_rotation import Rotation_COCOeval


def evaluate(dataset, predictions, result_file, score_threshold=None, epoch=0):
    coco_results = {}
    coco_results['bbox'] = make_coco_detection(predictions, dataset, score_threshold)

    results = COCOResult('bbox')

    path = os.path.join(result_file, str(epoch)+'_result.json')
    res = evaluate_predictions_on_coco(dataset.coco, coco_results['bbox'], path, 'bbox')
    results.update(res)
    # with tempfile.NamedTemporaryFile() as f:
    #     path = f.name
    #     res = evaluate_predictions_on_coco(
    #         dataset.coco, coco_results['bbox'], path, 'bbox'
    #     )
    #     results.update(res)

    print(results)

    return results.results


def evaluate_predictions_on_coco(coco_gt, results, result_file, iou_type):
    with open(result_file, 'w') as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(str(result_file)) if results else COCO()

    coco_eval = Rotation_COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.iouThrs = np.linspace(.25, 0.95, int(np.round((0.95 - .25) / .05)) + 1, endpoint=True)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    score_threshold = compute_thresholds_for_classes(coco_eval)


    return coco_eval


def compute_thresholds_for_classes(coco_eval):
    precision = coco_eval.eval['precision']
    precision = precision[0, :, :, 0, -1]
    scores = coco_eval.eval['scores']
    scores = scores[0, :, :, 0, -1]

    recall = np.linspace(0, 1, num=precision.shape[0])
    recall = recall[:, None]

    f1 = (2 * precision * recall) / (np.maximum(precision + recall, 1e-6))
    max_f1 = f1.max(0)
    max_f1_id = f1.argmax(0)
    scores = scores[max_f1_id, range(len(max_f1_id))]

    print('Maximum f1 for classes:')
    print(list(max_f1))
    print('Score thresholds for classes')
    print(list(scores))
    print('')
    return scores


def make_coco_detection(predictions, dataset, score_threshold=None):
    coco_results = []

    for id, pred in enumerate(predictions):
        orig_id = dataset.id2img[id]

        if len(pred) == 0:
            continue

        img_meta = dataset.get_image_meta(id)

        pred_resize = map_to_origin_image(img_meta, pred, flipmode='no', resize_mode='letterbox')

        boxes = pred_resize.bbox.tolist()
        scores = pred_resize.get_field('scores').tolist()
        labels = pred_resize.get_field('labels').tolist()


        labels = [dataset.id2category[i] for i in labels]
        if score_threshold is None:
            score_threshold = [0]*len(dataset.id2category)

        coco_results.extend(
            [
                {
                    'image_id': orig_id,
                    'category_id': labels[k],
                    'bbox': box,
                    'score': scores[k],
                }
                for k, box in enumerate(boxes)
                if scores[k] > score_threshold[labels[k] - 1]
            ]
        )

    return coco_results


class COCOResult:
    METRICS = {
        'bbox': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'segm': ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl'],
        'box_proposal': [
            'AR@100',
            'ARs@100',
            'ARm@100',
            'ARl@100',
            'AR@1000',
            'ARs@1000',
            'ARm@1000',
            'ARl@1000',
        ],
        'keypoints': ['AP', 'AP50', 'AP75', 'APm', 'APl'],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResult.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResult.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        return repr(self.results)


def map_to_origin_image(img_meta, pred, flipmode='no', resize_mode='letterbox'):
    '''
    img_meta: "id": int, "width": int, "height": int,"file_name": str,
    pred: boxlist object
    flipmode:'h':Horizontal flip,'v':vertical flip 'no': no flip
    resize_mode: 'letterbox' , 'wrap'
    '''

    assert pred.mode == 'xyxyxyxy'
    if flipmode == 'h':
        pred = pred.transpose(0)
    elif flipmode == 'v':
        pred = pred.transpose(1)
    elif flipmode == 'no':
        pass
    else:
        raise Exception("unspported flip mode, 'h', 'v' or 'no' ")

    width = img_meta['width']
    height = img_meta['height']

    resized_width, resized_height = pred.size

    if resize_mode == 'letterbox':
        if width > height:
            scale = resized_width / width
            size = (resized_width, int(scale * height))
        else:
            scale = resized_height / height
            size = (int(width * scale), resized_height)

        pred_resize = BoxList(pred.bbox, size, mode='xyxyxyxy')
        pred_resize._copy_extra_fields(pred)
        pred_resize = pred_resize.clip_to_image(remove_empty=True)
        pred_resize = pred_resize.resize((width, height))
        pred_resize = pred_resize.clip_to_image(remove_empty=True)
        #pred_resize = pred_resize.convert('xywh')

    elif resize_mode == 'wrap':
        pred_resize = pred.resize((width, height))
        pred_resize = pred_resize.convert('xyxyxyxy')
        pred_resize = pred_resize.clip_to_image(remove_empty=True)

    else:
        raise Exception("unspported reisze mode, either 'letterbox' or 'wrap' ")


    return pred_resize

