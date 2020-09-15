import json
import os
import glob
import cv2

annotations_info = {'images': [], 'annotations': [], 'categories': []}


CLASS_NAME = ('__background__','plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship',
                'tennis-court', 'basketball-court',
                'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor',
                'swimming-pool', 'helicopter')

categories_map = {class_name:i for i,class_name in enumerate(CLASS_NAME)}
image_root = '/Volumes/hy_mobile/03data/DOTA-v1.5/min_split_val'
json_annotation_path = '/Volumes/hy_mobile/03data/DOTA-v1.5/annotations/annotations_split_val.json'
image_file_paths = glob.glob(image_root+'/*.png')

ann_id = 1

for i, image_file_path in enumerate(image_file_paths):
    image_name = os.path.basename(image_file_path).split('.')[0]
    image_info = dict()
    image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    image_info = {'file_name': image_name+'.png', 'id': i + 1,
                  'height': height, 'width': width}
    annotations_info['images'].append(image_info)

    annotation_info = {"id": i+1, "image_id":i+1,
                       "bbox": [10, 30, 10, 60, 40, 30, 40, 60],
                       "category_id": 1,
                       "area": 900,
                       "iscrowd": 0}
    annotations_info['annotations'].append(annotation_info)


    ann_id += 1

with open(json_annotation_path, 'w') as f:
    json.dump(annotations_info, f, indent=4)

print('---整理后的标注文件---')
print('所有图片的数量：',  len(annotations_info['images']))
print('所有标注的数量：',  len(annotations_info['annotations']))
print('所有类别的数量：',  len(annotations_info['categories']))
