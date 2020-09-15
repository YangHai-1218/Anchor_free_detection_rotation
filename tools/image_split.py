import cv2
import glob
import os
import math
import numpy as np
import copy
from utils import BoxList,filter_bboxes
import torch
import random

image_root = '/Volumes/hy_mobile/03data/DOTA-v1.5/min_val'
anno_root = '/Volumes/hy_mobile/03data/DOTA-v1.5/val/min_labelTxt/'

split_image_dir = '/Volumes/hy_mobile/03data/DOTA-v1.5/min_split_val'
split_anno_dir = '/Volumes/hy_mobile/03data/DOTA-v1.5/min_anno_split_val'
class_name = ('__background__', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                'small-vehicle', 'large-vehicle', 'ship',
                'tennis-court', 'basketball-court',
                'storage-tank', 'soccer-ball-field',
                'roundabout', 'harbor',
                'swimming-pool', 'helicopter')
NAME_TAB = {name:i for i,name in enumerate(class_name)}
NAME_TAB_inverse = {i:name for i,name in enumerate(class_name)}

target_size = (768, 768) # width, height
bin_size = 256
min_size = (511, 511) # width, height
ratio = (1, 1) # sliding window : random sampled







def list_to_str(list_):
    list_ = [str(x)+' ' for x in list_]
    return "".join(list_)



if __name__ =='__main__':
    image_file_paths = glob.glob(image_root + '/*.png')
    for image_file_path in image_file_paths:
        image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
        image_name = os.path.basename(image_file_path).split('.')[0]

        anno_path = os.path.join(anno_root, image_name + '.txt')
        with open(anno_path) as f:
            lines = f.readlines()
            cls_names = []
            labels = []
            bboxes = []
            info = lines[:2]
            lines = lines[2:]
            for line in lines:
                bbox_info = line.split()
                bbox = bbox_info[0:8]
                bbox = [*map(lambda x: float(x), bbox)]
                #cls_name = bbox_info[0]
                cls_name = bbox_info[-2]
                bboxes.append(bbox)
                cls_names.append(cls_name)
                label = NAME_TAB[cls_name]
                labels.append(label)
        bboxes = np.array(bboxes)
        labels = torch.tensor(labels)
        #labels = np.array(cls_names)

        try:
            height, width,  _ = image.shape
        except:
            height, width = image.shape
        cols = list(range(0, width, target_size[0] - bin_size))  # 竖列 x
        rows = list(range(0, height, target_size[1] - bin_size))  # 横排 y


        random_col_row_w_h = []
        # random sample
        if len(cols) > len(rows) and int(len(cols)/ratio[0]) >= 1:
            col_sample_times = int(len(cols) / ratio[0])

            col_random_index = [random.choice(list(range(ratio[0])))+i*ratio[0] for i in range(col_sample_times)]
            cols_random = [cols[index] for index in col_random_index]

            for col in cols_random:
                for row in rows:
                    random_col = col + random.randint(10, bin_size)
                    random_row = row + random.randint(10, bin_size)
                    if (width - random_col) < min_size[0]:
                        continue
                    random_width = random.randint(min_size[0],
                                                  int(min(target_size[0]*1.5, width-random_col)))
                    if (height - random_row) < min_size[1]:
                        continue
                    random_height = random.randint(min_size[1],
                                                   int(min(target_size[1]*1.5, height-random_row)))
                    random_col_row_w_h.append((random_col,random_row,random_width,random_height))

        elif len(rows) > len(cols) and int(len(rows) / ratio[0]) >= 1:
            row_sample_times = int(len(rows) / ratio[0])
            row_random_index = [random.choice(list(range(ratio[0]))) + i*ratio[0] for i in range(row_sample_times)]
            rows_random = [rows[index] for index in row_random_index]
            for row in rows_random:
                for col in cols:
                    random_row = row + random.randint(10, bin_size)
                    random_col = col + random.randint(10, bin_size)
                    if (width - random_col) < min_size[0]:
                        continue
                    random_width = random.randint(min_size[0],
                                                  int(min(target_size[0]*1.5, width-random_col)))
                    if (height - random_row) < min_size[1]:
                        continue
                    random_height = random.randint(min_size[1],
                                                   int(min(target_size[1]*1.5, height-random_row)))
                    random_col_row_w_h.append((random_col, random_row, random_width, random_height))
        else:
            pass



        col_last_width = width - cols[-1]
        if col_last_width < min_size[0]:
            if col_last_width + target_size[0] - bin_size >= min_size[0] * 2:
                refined_col_width = math.ceil((width - cols[-2]) / 2)
                cols[-1] = math.ceil(cols[-2] + refined_col_width - bin_size/2)
            else:
                if len(cols) < 3:
                    refined_col_width = width
                    cols = [0]
                else:
                    refined_col_width = math.ceil((width - cols[-3]) / 3)
                    cols[-2] = math.ceil(cols[-3] + refined_col_width - bin_size/2)
                    cols[-1] = math.ceil(cols[-2] + refined_col_width - bin_size/2)

        row_last_height = height - rows[-1]
        if row_last_height < min_size[1]:
            if row_last_height + target_size[1] - bin_size >= min_size[1] * 2:
                refined_row_height = math.ceil((height - rows[-2]) / 2)
                rows[-1] = math.ceil(rows[-2] + refined_row_height - bin_size/2)
            else:
                if len(rows) < 3:
                    refined_row_height = height
                    rows = [0]
                else:
                    refined_row_height = math.ceil((height - rows[-3]) / 3)
                    rows[-2] = math.ceil(rows[-3] + refined_row_height - bin_size/2)
                    rows[-1] = math.ceil(rows[-2] + refined_row_height - bin_size/2)

        origin_bboxes = BoxList(bboxes, image_size=(width, height), mode='xyxyxyxy')
        origin_bboxes = origin_bboxes.change_order_to_clockwise()
        origin_bboxes.add_field('labels', labels)
        for i, col in enumerate(cols):
            for j, row in enumerate(rows):



                try:
                    image_width = cols[i+1] - col + bin_size
                except:
                    image_width = width - col

                try:
                    image_height = rows[j+1] - row + bin_size
                except:
                    image_height = height - row

                # cropped_image_name image_base_name|x_y_width_height
                cropped_image_name = f'{image_name}|{col}_{row}_{image_width}_{image_height}'

                try:
                    croped_image = image[row:row + image_height, col:col + image_width, :]
                except:
                    croped_image = image[row:row + image_height, col:col+image_width]
                cropped_image_path = os.path.join(split_image_dir, cropped_image_name + '.png')
                cv2.imwrite(cropped_image_path, croped_image)


                cropped_target = origin_bboxes.crop([col, row, col+image_width, row+image_height])
                cropped_target = filter_bboxes(origin_bboxes, cropped_target,
                                              [col, row, col+image_width, row+image_height])
                if len(cropped_target) > 0:
                    cropped_target = cropped_target.clip_to_image(remove_empty=True)

                cropped_bboxes = cropped_target.bbox

                cropped_labels = cropped_target.get_field("labels")


                cropped_bboxes = cropped_bboxes.tolist()
                cropped_labels = cropped_labels.tolist()

                anno = []
                #anno[:2] = info
                for bbox, label in zip(cropped_bboxes, cropped_labels):
                    label = NAME_TAB_inverse[label]
                    append_str = "".join([list_to_str(bbox), ' ', label])
                    append_str = append_str + '\n'
                    anno.append(append_str)

                cropped_anno_path = os.path.join(split_anno_dir, cropped_image_name + '.txt')
                with open(cropped_anno_path, 'w') as f:
                    f.writelines(anno)


        if random_col_row_w_h is not None:
            for col, row, image_width, image_height in random_col_row_w_h:
                cropped_image_name = f'{image_name}|{col}_{row}_{image_width}_{image_height}'

                try:
                    croped_image = image[row:row + image_height, col:col + image_width, :]
                except:
                    croped_image = image[row:row + image_height, col:col+image_width]

                cropped_image_path = os.path.join(split_image_dir, cropped_image_name + '.png')
                cv2.imwrite(cropped_image_path, croped_image)

                cropped_target = origin_bboxes.crop([col, row, col+image_width, row+image_height])
                cropped_target = filter_bboxes(origin_bboxes, cropped_target,
                                              [col, row, col+image_width, row+image_height])
                if len(cropped_target) > 0:
                    cropped_target = cropped_target.clip_to_image(remove_empty=True)
                cropped_bboxes = cropped_target.bbox
                cropped_labels = cropped_target.get_field("labels")
                cropped_bboxes = cropped_bboxes.tolist()
                cropped_labels = cropped_labels.tolist()

                anno = []
                # anno[:2] = info
                for bbox, label in zip(cropped_bboxes, cropped_labels):
                    label = NAME_TAB_inverse[label]
                    append_str = "".join([list_to_str(bbox), ' ', label])
                    append_str = append_str + '\n'
                    anno.append(append_str)

                cropped_anno_path = os.path.join(split_anno_dir, cropped_image_name + '.txt')
                with open(cropped_anno_path, 'w') as f:
                    f.writelines(anno)
