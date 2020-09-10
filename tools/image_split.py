import cv2
import glob
import os
import math
import numpy as np
import copy

image_root = '/Users/haiyang/Documents/03data/1-2'
anno_root = '/Users/haiyang/Documents/03data/1-2'

split_image_dir = '/Users/haiyang/Documents/03data/1-2'
split_anno_dir = '/Users/haiyang/Documents/03data/1-2'

target_size = (1024,1024) # width, height
bin_size = 256
min_size = (512,512) # width, height







def list_to_str(list_):
    list_ = [str(x)+' ' for x in list_]
    return "".join(list_)



if __name__ =='__main__':
    image_file_paths = glob.glob(image_root + '/*.tiff')
    for image_file_path in image_file_paths:
        image = cv2.imread(image_file_path,cv2.IMREAD_ANYDEPTH)
        image_name = os.path.basename(image_file_path).split('.')[0]

        anno_path = os.path.join(anno_root, image_name + '.txt')
        with open(anno_path,encoding='gbk') as f:
            lines = f.readlines()
            cls_names = []
            bboxes = []
            #info = lines[:2]
            #lines = lines[2:]
            for line in lines:
                bbox_info = line.split()
                bbox = bbox_info[1:9]
                bbox = [*map(lambda x: float(x), bbox)]
                cls_name = bbox_info[0]
                bboxes.append(bbox)
                cls_names.append(cls_name)
        bboxes = np.array(bboxes)
        labels = np.array(cls_names)

        try:
            height, width,  _ = image.shape
        except:
            height, width = image.shape
        cols = list(range(0, width, target_size[0] - bin_size))  # 竖列 x
        rows = list(range(0, height, target_size[1] - bin_size))  # 横排 y

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

        for i, col in enumerate(cols):
            for j, row in enumerate(rows):

                cropped_bboxes = copy.copy(bboxes)
                cropped_labels = copy.copy(labels)

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


                croped_image = image[row:row + image_height, col:col + image_width]
                cropped_image_path = os.path.join(split_image_dir, cropped_image_name + '.png')
                cv2.imwrite(cropped_image_path, croped_image)

                # x clip
                cropped_bboxes[:, 0:8:2] -= col
                cropped_bboxes[:, 0:8:2] = np.clip(cropped_bboxes[:, 0:8:2], 0, image_width).astype(np.float32)

                # y clip
                cropped_bboxes[:, 1:8:2] -= row
                cropped_bboxes[:, 1:8:2] = np.clip(cropped_bboxes[:, 1:8:2], 0, image_height).astype(np.float32)

                keep = (cropped_bboxes[:, 0:8:2].sum(axis=-1) > 0) & \
                       (cropped_bboxes[:, 0:8:2].sum(axis=-1) < 4*image_width) & \
                       (cropped_bboxes[:, 1:8:2].sum(axis=-1) > 0) & \
                       (cropped_bboxes[:, 1:8:2].sum(axis=-1) < 4*image_height)

                if keep.sum() > 0:
                    print(f'{cropped_image_name} has targets')

                cropped_bboxes = cropped_bboxes[keep].astype(np.int32).tolist()
                cropped_labels = cropped_labels[keep].tolist()

                anno = []
                #anno[:2] = info
                for bbox, label in zip(cropped_bboxes, cropped_labels):
                    append_str = "".join([list_to_str(bbox), ' ', label])
                    append_str = append_str + '\n'
                    anno.append(append_str)

                cropped_anno_path = os.path.join(split_anno_dir, cropped_image_name + '.txt')
                with open(cropped_anno_path, 'w') as f:
                    f.writelines(anno)

