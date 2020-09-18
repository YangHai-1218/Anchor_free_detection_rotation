import os

import torch
from torchvision import datasets

from utils.boxlist import BoxList
import cv2
import numpy as np
import random

def has_only_empty_bbox(annot):
    # if bbox width and height <=1 , then it is a empty box
    return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


def has_valid_annotation(annot):
    if len(annot) == 0:
        return False

    if has_only_empty_bbox(annot):
        return False

    return True


class DOTADataset(datasets.CocoDetection):


    NAME_TAB = ('__background__', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship',
                'tennis-court', 'basketball-court',
                'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor',
                'swimming-pool', 'helicopter')

    def __init__(self, path, split,image_folder_name, anno_folder_name,transform=None):
        """
        path : dataset folder path
               dataset structure:
            ├── dataset_path
            │   ├── annotations
            │   │   ├── anno_folder_name +'train'.json
            │   │   ├── anno_folder_name + 'val'.json
            │   │   ├── anno_folder_name + 'test'.json
            │   ├── image_folder_name+'train'
            │   ├── image_folder_name+'val'
            │   ├── image_folder_name+'test'
        """

        root, annot = self.get_root_annotation_path(path,split,image_folder_name,anno_folder_name)

        super().__init__(root, annot)

        self.ids = sorted(self.ids)

        if split == 'train' or split == 'val_loss':
            ids = []

            for id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=id, iscrowd=None)
                annot = self.coco.loadAnns(ann_ids)

                if has_valid_annotation(annot):
                    ids.append(id)

            self.ids = ids

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.id2img = {k: v for k, v in enumerate(self.ids)}

        self.transformer = transform

    def set_transform(self,transform):
        self.transformer = transform



    def get_root_annotation_path(self,path,split,image_folder_name,anno_folder_name):
        '''
        root : image dir
        annot: annotation file path
        '''
        self.split = split
        self.anno_folder_name = anno_folder_name
        self.image_folder_name = image_folder_name
        '''split: train, val, test'''

        if split == 'val_loss':
            annot = os.path.join(path, 'annotations', f"{self.anno_folder_name}val.json")
            root = os.path.join(path, f'{self.image_folder_name}val')
        else:
            annot = os.path.join(path, 'annotations', f'{self.anno_folder_name}{split}.json')
            root = os.path.join(path, f'{self.image_folder_name}{split}')
        return root, annot

    def __getitem__(self, index, transform_enable=True):

        if isinstance(index, tuple) or isinstance(index, list):
            transform_enable = index[1]
            index = index[0]
        else:
            transform_enable = True

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annots = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']


        img = cv2.imread(os.path.join(self.root, path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 2:
            # if single channel image, then convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            pass
        else:
            raise RuntimeError("{} channel image not supported".format(img.ndim))



        height, width,_ = img.shape
        annots = [o for o in annots if o['iscrowd'] == 0]

        boxes = [o['bbox'] for o in annots]
        boxes = torch.as_tensor(boxes).reshape(-1, 8)
        #target = BoxList(boxes, (width,height), mode='xyxyxyxy').convert('xywha')
        target = BoxList(boxes, (width,height), mode='xyxyxyxy')

        target = target.change_order_to_clockwise()
        target = target.convert('xywha_d')


        #target = target.convert('xywha')


        classes = [o['category_id'] for o in annots]
        classes = [self.category2id[c] for c in classes]
        classes = torch.tensor(classes)
        # target.fields['labels'] = classes
        target.add_field('labels', classes)


        target = target.clip_to_image(remove_empty=True)

        if self.transformer is not None and transform_enable:
            img, target = self.transformer(img, target)

        return img, target, index, path

    def get_image_meta(self, index):
        id = self.id2img[index]
        img_data = self.coco.imgs[id]

        return img_data







class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        if max_size[1] % stride != 0:
            max_size[1] = (max_size[1] | (stride - 1)) + 1
        if max_size[2] % stride != 0:
            max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)


def collate_fn(config):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], config.size_divisible)
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data
