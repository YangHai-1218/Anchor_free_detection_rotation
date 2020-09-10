import os
import cv2
import json
import xml.dom.minidom
import xml.etree.ElementTree as ET



CLASS_NAME=[
    '__background__',
    'golffield',
    'Expressway-toll-station',
    'vehicle',
    'trainstation',
    'chimney',
    'storagetank',
    'ship',
    'harbor',
    'airplane',
    'groundtrackfield',
    'tenniscourt',
    'dam',
    'basketballcourt',
    'Expressway-Service-area',
    'stadium',
    'airport',
    'baseballfield',
    'bridge',
    'windmill',
    'overpass',
]


data_dir = 'E:/01科研/03data/DIOR' #根目录文件，其中包含image文件夹和box文件夹（根据自己的情况修改这个路径）

image_file_dir = os.path.join(data_dir, 'JPEGImages-test')
xml_file_dir = os.path.join(data_dir, 'Annotations')
test_path = os.path.join(data_dir,'ImageSets/Main/test.txt')



annotations_info = {'images': [], 'annotations': [], 'categories': []}

categories_map = {class_name:i for i,class_name in enumerate(CLASS_NAME)}
del categories_map['__background__']

for key in categories_map:
    categoriy_info = {"id":categories_map[key], "name":key}
    annotations_info['categories'].append(categoriy_info)

# file_names = [image_file_name.split('.')[0]
#               for image_file_name in os.listdir(image_file_dir)]
ann_id = 1

with open(test_path,'r') as f:
    file_names = f.readlines()

for i in range(len(file_names)):
    file_names[i] = file_names[i].strip()


for i, file_name in enumerate(file_names):

    image_file_name = file_name + '.jpg'
    xml_file_name = file_name + '.xml'
    image_file_path = os.path.join(image_file_dir, image_file_name)
    xml_file_path = os.path.join(xml_file_dir, xml_file_name)

    image_info = dict()
    #image = cv2.cvtColor(cv2.imread(image_file_path), cv2.COLOR_BGR2RGB)
    #height, width, _ = image.shape


    DOMTree = xml.dom.minidom.parse(xml_file_path)
    collection = DOMTree.documentElement

    heights = collection.getElementsByTagName('height')
    heights =  [height.firstChild.data for height in heights]
    height = int(heights[0])
    widths = collection.getElementsByTagName('width')
    widths = [width.firstChild.data for width in widths]
    width = int(widths[0])
    image_info = {'file_name': image_file_name, 'id': i + 1,
                  'height': height, 'width': width}

    annotations_info['images'].append(image_info)

    names = collection.getElementsByTagName('name')
    names = [name.firstChild.data for name in names]


    xmins = collection.getElementsByTagName('xmin')
    xmins = [xmin.firstChild.data for xmin in xmins]
    ymins = collection.getElementsByTagName('ymin')
    ymins = [ymin.firstChild.data for ymin in ymins]
    xmaxs = collection.getElementsByTagName('xmax')
    xmaxs = [xmax.firstChild.data for xmax in xmaxs]
    ymaxs = collection.getElementsByTagName('ymax')
    ymaxs = [ymax.firstChild.data for ymax in ymaxs]

    object_num = len(names)

    for j in range(object_num):
        if names[j] in categories_map:
            image_id = i + 1
            x1,y1,x2,y2 = int(xmins[j]),int(ymins[j]),int(xmaxs[j]),int(ymaxs[j])
            x1,y1,x2,y2 = x1 - 1,y1 - 1,x2 - 1,y2 - 1

            if x2 == width:
                x2 -= 1
            if y2 == height:
                y2 -= 1

            x,y = x1,y1
            w,h = x2 - x1 + 1,y2 - y1 + 1
            category_id = categories_map[names[j]]
            area = w * h
            annotation_info = {"id": ann_id, "image_id":image_id,
                               "bbox":[x, y, w, h], "category_id": category_id,
                               "area": area,"iscrowd": 0}
            annotations_info['annotations'].append(annotation_info)
            ann_id += 1

with  open('./test_annotations.json', 'w')  as f:
    json.dump(annotations_info, f, indent=4)


print('---整理后的标注文件---')
print('所有图片的数量：',  len(annotations_info['images']))
print('所有标注的数量：',  len(annotations_info['annotations']))
print('所有类别的数量：',  len(annotations_info['categories']))