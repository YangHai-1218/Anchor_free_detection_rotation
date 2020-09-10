import json
import os
import cv2
import glob
import numpy as np

annotations_info = {'images': [], 'annotations': [], 'categories': []}


CLASS_NAME = ('__background__','plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship',
                'tennis-court', 'basketball-court',
                'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor',
                'swimming-pool', 'helicopter')
categories_map = {class_name:i for i,class_name in enumerate(CLASS_NAME)}
del categories_map['__background__']
for key in categories_map:
    categoriy_info = {"id":categories_map[key], "name":key}
    annotations_info['categories'].append(categoriy_info)

image_root = '/Users/haiyang/Documents/03data/DOTA-v1.5/val/images/images'
annotation_root = '/Users/haiyang/Documents/03data/DOTA-v1.5/val/labelTxt-v1.0/labelTxt'
json_annotation_path = './val_annotations.json'

image_file_paths = glob.glob(image_root+'/*.png')
annotation_file_paths = glob.glob(annotation_root+'/*.txt')

ann_id = 1

for i,image_file_path in enumerate(image_file_paths):
    image_name = os.path.basename(image_file_path).split('.')[0]
    image_info = dict()
    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image_info = {'file_name': image_name+'.png', 'id': i + 1,
                  'height': height, 'width': width}
    annotations_info['images'].append(image_info)

    annotation_file_path = os.path.join(annotation_root, image_name+'.txt')
    with open(annotation_file_path) as f:
        lines = f.readlines()
        gsd = lines[1].split(':')[-1]
        lines = lines[2:]
        for line in lines:
            bbox_info = line.split()
            bbox = bbox_info[:8]
            bbox = [*map(lambda x: float(x), bbox)]
            cls_name = bbox_info[8]
            category_id = categories_map[cls_name]

            rbbox = cv2.minAreaRect(np.array(bbox).reshape((4,2)).astype(np.float32))
            area = rbbox[1][0] * rbbox[1][1]
            annotation_info = {"id": ann_id, "image_id": i+1,
                               "bbox": bbox, "category_id": category_id,
                               "area": area, "iscrowd": 0}

            annotations_info['annotations'].append(annotation_info)

            ann_id += 1

with open(json_annotation_path, 'w') as f:
    json.dump(annotations_info, f, indent=4)

print('---整理后的标注文件---')
print('所有图片的数量：',  len(annotations_info['images']))
print('所有标注的数量：',  len(annotations_info['annotations']))
print('所有类别的数量：',  len(annotations_info['categories']))




