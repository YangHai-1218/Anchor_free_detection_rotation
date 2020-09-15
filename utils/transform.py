import random

import numpy as np
from PIL import ImageDraw
import torch
from torchvision.transforms import functional as F
import cv2
import copy
import math
from .boxlist import BoxList,boxlist_ioa,cat_boxlist,filter_bboxes

"""
Note this: this data augemenation pipeline is based on cv2, 
so if you want to use the following methods, 
make sure the input data is np.ndarray, and BGR format
"""


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, img_size):
        # 短边resize，keep ratio
        w, h = img_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_orig = float(min((w, h)))
            max_orig = float(max((w, h)))

            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)

        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, img, target):
        size = self.get_size(img.size)
        img = F.resize(img, size)
        target = target.resize(img.size)

        return img, target

class Crop:
    def __call__(self, img, target):
        img = img[100:500,100:500]
        target = target.crop([100,100,500,500])
        target = target.clip_to_image()
        return img, target
class Resize_For_Efficientnet:
    """ similar letter box resize implement"""

    def __init__(self, compund_coef=0, color=(114, 114, 114)):
        self.target_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.target_size = self.target_sizes[compund_coef]
        self.color = color

    def __call__(self, img, target):
        # 长边resize到target_size,短边成比例缩放，之后进行pad
        size = (self.target_size,) * 2
        height, width, _ = img.shape
        if height > width:
            scale = self.target_size / height
            resized_height = self.target_size
            resized_width = int(width * scale)
        else:
            scale = self.target_size / width
            resized_height = int(height * scale)
            resized_width = self.target_size

        resized_image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        new_image[:, :, 0] = self.color[0]
        new_image[:, :, 1] = self.color[1]
        new_image[:, :, 2] = self.color[2]
        new_image[0:resized_height, 0:resized_width] = resized_image

        resized_target = target.resize((resized_width, resized_height))
        target = BoxList(resized_target.bbox, image_size=(self.target_size,) * 2, mode=resized_target.mode)
        target._copy_extra_fields(resized_target)
        return new_image, target


class RandomResize:
    def __init__(self, min_size_range, max_size):
        self.min_size_range = min_size_range
        self.max_size = max_size

    def __call__(self, img, target):
        assert (len(self.min_size_range) == 2)
        min_size = random.randint(self.min_size_range[0], self.min_size_range[1])
        return Resize(min_size, self.max_size)(img, target)


class RandomScale:
    def __init__(self, min_scale, max_scale):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img, target):
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        h *= scale
        w *= scale
        size = (round(h), round(w))

        img = F.resize(img, size)
        target = target.resize(img.size)

        return img, target


class RandomBrightness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img, target):
        factor = random.uniform(-self.factor, self.factor)
        img = F.adjust_brightness(img, 1 + factor)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = np.fliplr(img)
            img = np.ascontiguousarray(img)
            if target is not None:
                target = target.transpose(0)

        return img, target


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = np.flipud(img)
            img = np.ascontiguousarray(img)
            target = target.transpose(1)
        return img, target


class RandomRotate:
    def __init__(self, p=0.5, rotate_time=1):
        self.p = p
        self.rotate_time = rotate_time

    def __call__(self, img, target):
        if random.random() < self.p:
            random_rotate_time = random.choice(list(range(1, self.rotate_time+1)))
            for i in range(random_rotate_time):
                img = np.rot90(img)
                target = target.rotate_90()
                img = np.ascontiguousarray(img)
        return img, target


class ToTensor:
    def __call__(self, img, target):
        if img.dtype == np.uint16:
            div_num = 2**16-1
        elif img.dtype == np.uint8:
            div_num = 2**8-1
        elif img.dtype == np.uint32:
            div_num = 2**32-1
        elif img.dtype == np.uint64:
            div_num = 2**64-1
        else:
            raise RuntimeError("Not supported data type :{}".format(img.dtype))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, channel last to channel first
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        return img.float().div(div_num), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target





class RandomMixUp:
    '''
    https://github.com/dmlc/gluon-cv/blob/49be01910a8e8424b017ed3df65c4928fc918c67/gluoncv/data/mixup/detection.py
    '''
    # TODO WEIGHTLOSS
    def __init__(self,dataset,alpha=1.5,beta=1.5,p=0.5):
        self.alpha = alpha
        self.beta = beta
        self.p = p
        self.dataset = dataset

    def __call__(self, img, target):
        if random.random()>self.p:
            weight = target.bbox.new_ones(len(target))
            target.add_field("weights",weight)
            return img, target
        lambd = np.random.beta(self.alpha, self.beta)
        img1 = img
        target1 = target
        idx2 = random.choice(range(len(self.dataset)))
        img2,target2,_ = self.dataset[(idx2,False)]
        height = max(img1.shape[0],img2.shape[0])
        width = max(img1.shape[1],img2.shape[1])

        new_image = np.zeros((height,width,3),dtype=np.float32)
        new_image[0:img1.shape[0],0:img1.shape[1],:] = img1 * lambd
        new_image[0:img2.shape[0],0:img2.shape[1],:] += img2 * (1-lambd)
        weight1 = torch.zeros(len(target1),dtype=torch.float) + lambd
        weight2 = torch.zeros(len(target2),dtype=torch.float) + (1-lambd)

        target1.add_field('weights',weight1)
        target2.add_field('weights',weight2)
        target1.size = (width,height)
        target2.size = (width,height)
        target = cat_boxlist([target1,target2])
        new_image = new_image.astype(np.uint8)
        return new_image,target




class Multi_Scale_with_Crop:
    def __init__(self,scales,target_size,pad_color=(114, 114, 114)):
        '''
        scales: list, short edge of the image is randomly sample from scales, and the long edge is fixed as 3840
        target_size : after rescaling the image, random crop a region of target size from rescaled image
        target_size: (w,h)
        '''
        self.short_edges = scales
        self.long_edge = 3840
        self.target_size = target_size
        self.color = pad_color

    def __call__(self, img, target):
        short_edge = random.choice(self.short_edges)
        height, width, _ = img.shape
        if height > width:
            scale_factor = short_edge / width
            resized_height = math.ceil(scale_factor*height)
            resized_width = short_edge
            if resized_height > self.long_edge:
                scale_factor = self.long_edge / height
                resized_width = math.ceil(scale_factor*width)
                resized_height = self.long_edge
        else:
            scale_factor = short_edge/height
            resized_width = math.ceil(scale_factor*width)
            resized_height = short_edge
            if resized_width > self.long_edge:
                scale_factor = self.long_edge / width
                resized_height = math.ceil(scale_factor*height)
                resized_width = self.long_edge

        #print('resized_width:',resized_width,'resized_height:',resized_height)
        resized_image = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        resized_target = target.resize((resized_width, resized_height))


        pad_w = self.target_size[0] - resized_width if self.target_size[0] > resized_width else 0
        pad_h = self.target_size[1] - resized_height if self.target_size[1] > resized_height else 0
        new_image = np.zeros((resized_height+pad_h,resized_width+pad_w, 3), dtype=np.uint8)
        new_image[:, :, 0] = self.color[0]
        new_image[:, :, 1] = self.color[1]
        new_image[:, :, 2] = self.color[2]
        new_image[0:resized_height, 0:resized_width] = resized_image


        #determine crop area
        left_x = resized_width + pad_w
        top_y = resized_height + pad_h
        right_x = left_x + self.target_size[0]
        bottom_y = top_y + self.target_size[1]


        repeat_max_time = 20
        repeat_time = 0
        while (resized_target.bbox[:, 0] - 30 - left_x).sum() <= 0 \
            or (resized_target.bbox[:, 1] - 30 - top_y).sum() <= 0 \
            or (resized_target.bbox[:, 0] - right_x).sum() >= 0 \
            or (resized_target.bbox[:, 1] - bottom_y).sum() >= 0 :

            if repeat_time >=repeat_max_time:
                break
            left_x = random.randint(0, resized_width + pad_w-self.target_size[0])
            top_y = random.randint(0, resized_height + pad_h-self.target_size[1])
            right_x = left_x + self.target_size[0]
            bottom_y = top_y + self.target_size[1]

            repeat_time += 1

        if repeat_time >= repeat_max_time:
            croped_image = cv2.resize(new_image, self.target_size, interpolation=cv2.INTER_LINEAR)
            croped_target = resized_target.resize(self.target_size)
            return croped_image, croped_target

        croped_image = new_image[top_y:bottom_y, left_x:right_x]
        croped_target = resized_target.crop([left_x, top_y, right_x,bottom_y])

        try:
            croped_target = filter_bboxes(resized_target, croped_target, [left_x,top_y,right_x,bottom_y],
                                            keep_best_ioa=True)
        except:
            croped_image = cv2.resize(new_image, self.target_size, interpolation=cv2.INTER_LINEAR)
            croped_target = resized_target.resize(self.target_size)
            return croped_image, croped_target


        croped_target = croped_target.clip_to_image(remove_empty=True)

        return croped_image, croped_target







class Cutout:
    """
    https://arxiv.org/abs/1708.04552
    """
    def __init__(self,p=0.5):
        self.p = p


    def __call__(self, img, target):
        if random.random() > self.p:
            return img,target


        height,width,_ = img.shape
        scales = [0.25] * 1 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16

        for s in scales:
            mask_h = random.randint(1, int(height * s))
            mask_w = random.randint(1, int(width* s))

            # box
            xmin = max(0, random.randint(0, width) - mask_w // 2)
            ymin = max(0, random.randint(0, height) - mask_h // 2)
            xmax = min(width, xmin + mask_w)
            ymax = min(height, ymin + mask_h)
            if xmin == xmax or ymin == ymax:
                continue


            target_bak = copy.deepcopy(target)
            mask_box = torch.tensor([xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin])[None, :]
            mask_box = BoxList(mask_box, image_size=(width, height), mode='xyxyxyxy')
            ioa = boxlist_ioa(mask_box, target).squeeze()
            target = target[(ioa < 0.6).reshape(-1)]
            if len(target) <= 0:
                return img, target_bak
            else:
                # apply random color mask
                img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        return img, target

class Mosaic:
    """ https://arxiv.org/abs/1905.04899
    """
    def __init__(self,image_size,dataset):
        self.image_size = image_size # width,height
        self.dataset = dataset

    def __call__(self, img, target):
        labels4 = []
        img4 = np.full((self.image_size[1] * 2, self.image_size[0]* 2, img.shape[2]), 114, dtype=np.uint8)
        xc = int(random.uniform(self.image_size[0] * 0.5, self.image_size[0] * 1.5))
        yc = int(random.uniform(self.image_size[1] * 0.5, self.image_size[1] * 1.5))
        other_indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        for i in range(4):

            if i == 0: # left top
                h, w, _ = img.shape
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h# xmin, ymin, xmax, ymax (small image)
                x_left,y_top = x1a,y1a
            elif i == 1: # right top
                img,target,_ = self.dataset[(other_indices[i-1],False)]
                h,w,_ = img.shape
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.image_size[0] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # left bottom
                img, target,_ = self.dataset[(other_indices[i - 1],False)]
                h, w, _ = img.shape
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.image_size[1] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                img, target , _ = self.dataset[(other_indices[i - 1],False)]
                h, w, _ = img.shape
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.image_size[0] * 2), min(self.image_size[1] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                x_right,y_bottom = x2a,y2a

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            if len(target) > 0:
                assert target.mode == 'xywha' or target.mode == 'xywha_d'
                target.bbox[:, 0] = target.bbox[:, 0] + padw
                target.bbox[:, 1] = target.bbox[:, 1] + padh
                target.size = self.image_size
            labels4.append(target)


        labels4 = cat_boxlist(labels4)
        # image_box = [self.image_size[0] // 2, self.image_size[1] // 2,
        #              self.image_size[0] // 2 + self.image_size[0], self.image_size[1] // 2 + self.image_size[1]]
        image_box = [x_left, y_top, x_right, y_bottom]
        cropped_labels4 = labels4.crop(image_box)
        cropped_labels4 = filter_bboxes(labels4, cropped_labels4, image_box)
        cropped_labels4 = cropped_labels4.clip_to_image(remove_empty=True)

        img4 = img4[image_box[1]:image_box[3], image_box[0]:image_box[2], :]


        print(img4.shape)
        print(cropped_labels4.size)
        return img4, cropped_labels4







class RandomHSV:
    """
     https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/datasets.py
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, img, target):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        img_dst = copy.deepcopy(img)
        hue, sat, val = cv2.split(cv2.cvtColor(img_dst, cv2.COLOR_BGR2HSV))
        dtype = img_dst.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img_dst)
        return img_dst, target


class RandomAffine:
    '''
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    '''

    def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, border=0, fill_color=(114, 114, 114)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.border = border
        self.fill_color = fill_color

    def __call__(self, img, target):

        height = img.shape[0] + self.border * 2
        width = img.shape[1] + self.border * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-self.translate, self.translate) * img.shape[0] + self.border  # x translation (pixels)
        T[1, 2] = random.uniform(-self.translate, self.translate) * img.shape[1] + self.border  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (self.border != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=self.fill_color)

        # Target transform
        n = len(target)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))  # [x,y,1] to perform [x,y,1]*rotation martix ^ T
            xy[:, :2] = target.bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            boxes = torch.from_numpy(xy).reshape(-1, 4)

            target_new = BoxList(boxes, (width, height), mode='xyxy')
            target_new._copy_extra_fields(target)

            # reject warped points outside of image
            target_new = target_new.clip_to_image(remove_empty=False)

            w = target_new.bbox[:, 2] - target_new.bbox[:, 0]
            h = target_new.bbox[:, 3] - target_new.bbox[:, 1]
            area = w * h
            area0 = (target.bbox[:, 2] - target.bbox[:, 0]) * (target.bbox[:, 3] - target.bbox[:, 1])
            ar, _ = torch.max(torch.stack([w / (h + 1e-16), h / (w + 1e-16)], dim=1), dim=-1)
            # ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            keep = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

            target_new = target_new[keep]
        print(f'new shape:{img.shape}')
        return img, target_new


def test():
    from .dataset import DOTADataset
    from tools.visualize import draw_ploygon_bbox_text,COLOR_TABLE
    from torchvision import transforms

    dataset = DOTADataset('data/', split='train', image_folder_name='min_split_',
                          anno_folder_name='annotations_split_')
    # dataset = DOTADataset('/Volumes/hy_mobile/03data/DOTA-v1.5', split='train', image_folder_name='min_split_',
    #                       anno_folder_name='annotations_split_')

    transform = Compose(
        [
            RandomHSV(0.1,0.1,0.1),
            # RandomAffine(degrees= 1.98*1,translate=0.05 * 0,scale=0.1,shear=0.641 * 0),
            RandomHorizontalFlip(1),
            RandomVerticalFlip(1),
            RandomRotate(1, rotate_time=4),
            #RandomMixUp(dataset),
            Cutout(0.9),
            #Resize_For_Efficientnet(compund_coef=2),
            #Mosaic(image_size=(768,768),dataset=dataset),
            # Resize_For_Efficientnet(compund_coef=2),
            Multi_Scale_with_Crop(scales=[768, 896, 1024, 1152],target_size=(768, 768)),
            ToTensor(),
            # Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    def plot_targets_PIL(image, targets, dataset):
        targets = targets.convert('xyxyxyxy')
        draw = ImageDraw.Draw(image)
        labels = targets.get_field("labels")
        #weights = targets.get_field("weights")


        for target,label in zip(targets.bbox,labels):
            draw_ploygon_bbox_text(draw, target, dataset.NAME_TAB[label.item()], COLOR_TABLE[label.item()])


    def plot_targets_cv2(image, targets):
        for target, label in zip(targets.bbox, targets.get_field('labels')):
            cv2.rectangle(image, (int(target[0]), int(target[1])), (int(target[2]), int(target[3])), (0, 255, 0), 2)
            cv2.putText(image, dataset.NAME_TAB[int(label.item())], (int(target[0]), int(target[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image

    for i in range(0, 400):
        #i = random.choice(list(range(0,len(dataset))))
        img_origin, targets_origin, _ = dataset[i]

        # img_origin = plot_targets(img_origin, targets_origin)
        img_new, targets_new = transform(img_origin, targets_origin)
        if len(targets_new) == 0:
            breakpoint = 1
        # print(f'image_size:{img_new.size()}')
        print(f'new_targets:{targets_new}')
        # print(f'original_targets:{targets_origin}')

        # cv2.imshow('image_origin', img_origin)
        # cv2.waitKey()

        # cv2.imshow('image', img)
        # cv2.waitKey()
        # #img_combine = np.hstack((img,img_origin))


        print(targets_new)
        # img_new = transforms.ToPILImage()(img_new).convert('RGB')
        # plot_targets_PIL(img_new, targets_new, dataset)
        # img_new.show()
        breakpoint =1


if __name__ == '__main__':
    test()