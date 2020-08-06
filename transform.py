import random

import numpy as np
from PIL import Image
import torch
from torch.nn.functional import pad
from torchvision.transforms import functional as F
import cv2
import copy
import math
from boxlist import BoxList

"""
Note that: this data augemntation pipeline is based on opencv, 
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


class Resize_For_Efficientnet:
    """ similar letter box resize implement"""
    def __init__(self,compund_coef=0,color=(114,114,114)):
        self.target_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.target_size = self.target_sizes[compund_coef]
        self.color = color
    def __call__(self, img, target):
        # 长边resize到target_size,短边成比例缩放，之后进行pad
        size = (self.target_size,)*2
        height,width,_ = img.shape
        if height>width:
            scale = self.target_size / height
            resized_height = self.target_size
            resized_width = int(width * scale)
        else:
            scale = self.target_size / width
            resized_height = int(height * scale)
            resized_width = self.target_size

        resized_image = cv2.resize(img,(resized_width,resized_height),interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.target_size, self.target_size, 3),dtype=np.uint8)
        new_image[:,:,0] = self.color[0]
        new_image[:,:,1] = self.color[1]
        new_image[:,:,2] = self.color[2]
        new_image[0:resized_height, 0:resized_width] = resized_image

        resized_target = target.resize((resized_width, resized_height))
        target = BoxList(resized_target.bbox,image_size=(self.target_size,)*2,mode='xyxy')
        target._copy_extra_fields(resized_target)
        return new_image,target


class RandomResize:
    def __init__(self, min_size_range, max_size):
        self.min_size_range = min_size_range
        self.max_size = max_size

    def __call__(self, img, target):
        assert(len(self.min_size_range) == 2)
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
            target = target.transpose(0)

        return img, target

class RandomVerticalFlip:
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, img,target):
        if random.random()<self.p:
            img = np.flipud(img)
            img = np.ascontiguousarray(img)
            target = target.transpose(1)
        return img,target


class ToTensor:
    def __call__(self, img, target):
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, channel last to channel first
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        return img.float().div(255), target

    
class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target





class RandomBrightness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img, target):
        factor = random.uniform(-self.factor, self.factor)
        img = F.adjust_brightness(img, 1 + factor)

        return img, target


class RandomHSV:
    """
    from https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/datasets.py
    """
    def __init__(self,hgain=0.5,sgain=0.5,vgain=0.5):
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
        return img_dst,target

class RandomAffine:
    '''
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    '''
    def __init__(self,degrees=10, translate=.1, scale=.1, shear=10,border=0, fill_color=(114,114,114)):
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
        print(s)
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
            xy = np.ones((n * 4, 3)) # [x,y,1] to perform [x,y,1]*rotation martix ^ T
            xy[:, :2] = target.bbox[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            boxes = torch.from_numpy(xy).reshape(-1,4)

            target_new = BoxList(boxes, (height, width), mode='xyxy')
            target_new.add_field('labels',target.get_field('labels'))
            # reject warped points outside of image
            target_new = target_new.clip_to_image(remove_empty=False)


            w = target_new.bbox[:, 2] - target_new.bbox[:, 0]
            h = target_new.bbox[:, 3] - target_new.bbox[:, 1]
            area = w * h
            area0 = (target.bbox[:, 2] - target.bbox[:, 0]) * (target.bbox[:, 3] - target.bbox[:, 1])
            ar,_ = torch.max(torch.stack([w / (h + 1e-16), h / (w + 1e-16)],dim=1),dim=-1)
            #ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            keep = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)


            target_new = target_new[keep]

        return img,target_new

def test():
    from dataset import COCODataset
    from coco_meta import CLASS_NAME
    dataset = COCODataset('../../03data/coco2017/','val')



    def plot_targets(image,targets):
        for target,label in zip(targets.bbox,targets.get_field('labels')):
            cv2.rectangle(image, (int(target[0]),int(target[1])), (int(target[2]),int(target[3])), (0, 255, 0), 2)
            cv2.putText(image,CLASS_NAME[int(label.item())],(int(target[0]),int(target[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255), 2)
        return image
    for i in range(0,100):
        img_origin, targets_origin, _ = dataset[i]


        # randomhsv = RandomHSV(0.5,0.5,0.5)
        # img,_ = randomhsv(img_origin,targets)
        # randomaffine = RandomAffine(degrees=1.98,translate=0.05 * 0,scale=0.8,shear=0.641 * 0)
        # img,targets_new = randomaffine(img_origin,targets)
        # resize = Resize_For_Efficientnet(2)
        # img,targets_new = resize(img_origin,targets)
        # flip = RandomHorizontalFlip()
        # img,targets_new = flip(img_origin,targets)
        # flip = RandomVerticalFlip()
        # img_new,targets_new = flip(img_origin,targets_origin)
        transform = Compose(
            [
            # RandomHSV(0.1,0.1,0.1),
            # RandomAffine(degrees= 1.98*0,translate=0.05 * 0,scale=0.1,shear=0.641 * 0),
            #RandomHorizontalFlip(),
            Resize_For_Efficientnet(compund_coef=2),
            # ToTensor(),
            # Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])

        #img_origin = plot_targets(img_origin, targets_origin)
        img_new, targets_new = transform(img_origin, targets_origin)

        # cv2.imshow('image_origin', img_origin)
        # cv2.waitKey()
        img = plot_targets(img_new,targets_new)
        cv2.imshow('image', img)
        cv2.waitKey()
        # #img_combine = np.hstack((img,img_origin))
        #
        # from torchvision import transforms
        #
        #
        # print(targets_new)
        # img_new = transforms.ToPILImage()(img_new).convert('RGB')
        # img_new.show()




if __name__ == '__main__':
    test()












