import json
import os
import copy
from collections import defaultdict
from pycocotools.coco import COCO
from utils import BoxList, cat_boxlist, boxlist_ml_rnms
import torch
det_anno_path = ''
gt_anno_path = ''
nms_thresh = 0.3
save_dir = ''

def load_det(annotations):
    image_anno = defaultdict(list)
    annotations_ = copy.deepcopy(annotations)
    for annotation in annotations_:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        score = annotation['score']
        image_anno[image_id].append({'category_id':category_id,'bbox':bbox,'score':score})
    return image_anno


def convert_to_boxlist(annotations):
    boxlist_annotations = {}
    for image_id in annotations:
        bbox = [bbox_annotation['bbox'] for bbox_annotation in annotations[image_id]]
        score = [bbox_annotation['score'] for bbox_annotation in annotations[image_id]]
        label = [bbox_annotation['category_id'] for bbox_annotation in annotations[image_id]]
        boxlist = BoxList(bbox,image_size=(1024,1024),mode='xyxyxyxy')
        boxlist.add_field('scores', torch.as_tensor(score).reshape(-1))
        boxlist.add_field('labels', torch.as_tensor(label).reshape(-1))
        boxlist_annotations[image_id] = boxlist

    return boxlist_annotations

def map_to_origin_image(boxlist_annotations, cocogt):
    origin_image_anno = defaultdict(list)
    for image_id in boxlist_annotations:
        image_content = cocogt.loadImgs(ids= [image_id])
        image_content = image_content[0]
        image_name = image_content["file_name"]
        image_base_name = image_name.split('|')[0]
        row, col = image_name.split('|')[1].split('_')[:2] # col 竖列(x) row 横排(y)
        det_result = boxlist_annotations[image_id]
        offset = torch.tensor([col, row], dtype= torch.float32).repeat(4)
        det_result.bbox = det_result.bbox + offset
        det_result.size = (image_content['wdith'], image_content['height'])
        origin_image_anno[image_base_name].append(det_result)

    mapped_image_anno = {}
    for image_name in origin_image_anno:
        mapped_anno = cat_boxlist(origin_image_anno[image_name])
        mapped_image_anno[image_name] = mapped_anno

    return mapped_image_anno


def nms_for_every_image(mapped_annotations):
    nms_result = {}
    for image_name in mapped_annotations:
        nms_boxlist = boxlist_ml_rnms(mapped_annotations[image_name], nms_thresh)
        nms_result[image_name] = nms_boxlist
    return nms_result


def list_to_str(list_):
    list_ = [str(x)+' ' for x in list_]
    return "".join(list_)

def write_to_txt(result,cocogt):
    for image_name in result:
        det_boxlist = result[image_name]
        bboxes = det_boxlist.bbox.tolist()
        scores = det_boxlist.get_field('scores').tolist()
        labels = det_boxlist.get_field('labels').tolist()
        categories = [cocogt.loadCats(ids=label)[0] for label in labels]

        anno = []
        for bbox, category, score in zip(bboxes, categories, scores):
            append_str = "".join([category, ' ', list_to_str(bbox)])
            append_str = append_str + '\n'
            anno.append(append_str)
        save_path = os.path.join(save_dir, image_name+'.txt')
        with open(save_path,'w') as f:
            f.writelines(anno)


if __name__ == '__main__':
    det_result = json.load(det_anno_path)
    cocogt = COCO(gt_anno_path)
    annotations = load_det(det_result)
    annotations = convert_to_boxlist(annotations)
    mapped_annotations = map_to_origin_image(annotations,cocogt)
    nms_result = nms_for_every_image(mapped_annotations)
    write_to_txt(nms_result)





