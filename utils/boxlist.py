import torch
import cv2
import numpy as np
import math
import copy
from ops import batched_rnms, rnms
from ops import polygon_iou

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1

class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx5 Tensor,
    if mode is 'xyxyxyxy' , then the bounding boxes are represented as a Nx8 tensor,
        and the tensor means the four point of bounding box
    if mode is 'xywha', then the bounding boxes are reppresented as a Nx5 tensor,
        and the tensor means xc,yc,w,h,theta(radian)
    if mode is 'xywha_d', then the bounding boxes are reperesented as a Nx5 tensor,
        and the tensor means xc,yc,w,h,theta(degree)
    What's the difference between 'xyxyxyxy' and 'xywha', 'xyxyxyxy' may not represent a rectangle,
    but 'xywha' represents a rotated rectangle

    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, image_size, mode="xywha"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )

        if mode not in ("xywha", "xyxyxyxy", "xywha_d"):
            raise ValueError("mode should be 'xywha' , 'xyxyxyxy' or 'xywha_d' ")

        if bbox.size(-1) != 5 and (mode == 'xywha' or mode == 'xywha_d'):
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 5 for '{}' mode , got {}".format(mode,bbox.size(-1))
            )
        if bbox.size(-1) != 8 and mode == 'xyxyxyxy':
            raise ValueError(
                "last dimension of bbox should have a"
                " size of 8 for 'xyxyxyxy' mode, got {}".format(bbox.size(-1))
            )

        self.bbox = bbox
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}
        self.device = device



    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxyxyxy", "xywha", "xywha_d"):
            raise ValueError("mode should be 'xyxyxyxy', 'xywha' or 'xywha_d'")
        if mode == self.mode:
            return self
        if mode == 'xyxyxyxy':
            if self.mode == 'xywha':
                return self.xywha_to_xyxyxyxy()
            elif self.mode == 'xywha_d':
                return self.xywhad_to_xyxyxyxy()
            else:
                raise RuntimeError("Should not be here")
        elif mode == 'xywha':
            if self.mode == 'xyxyxyxy':
                return self.xyxyxyxy_to_xywha()
            elif self.mode == 'xywha_d':
                return self.xywhad_to_xywha()
            else:
                raise RuntimeError("Should not be here")
        elif mode == 'xywha_d':
            if self.mode == 'xywha':
                return self.xywha_to_xywhad()
            elif self.mode == 'xyxyxyxy':
                return self.xyxyxyxy_to_xywhad()
            else:
                raise RuntimeError("Should not be here")
        else:
            raise RuntimeError("Should not be hare")

    def change_order_to_clockwise(self):
        '''change the points order to clockwise order, [left_top,left_bottom,right_bottom,right_top]'''
        if self.mode == 'xywha' or self.mode == 'xywha_d':
            print("'{} mode don't have to change point order".format(self.mode))
            return self

        # clockwise_bbox = []
        # for bbox in self.bbox:
        #     clockwise_points = []
        #
        #     p1, p2, p3, p4 = bbox[:2], bbox[2:4], bbox[4:6], bbox[6:8]
        #     points = torch.stack([p1, p2, p3, p4], dim=0)
        #     x = torch.tensor([p1[0], p2[0], p3[0], p4[0]])
        #     _, index = x.sort()
        #     left_points = points[index[:2]]
        #     right_points = points[index[2:]]
        #     if left_points[0, 1] < left_points[1, 1]:
        #         clockwise_points.extend([left_points[0, :], left_points[1, :]])
        #     else:
        #         clockwise_points.extend([left_points[1, :], left_points[0, :]])
        #
        #     if right_points[0, 1] > right_points[1, 1]:
        #         clockwise_points.extend([right_points[0, :], right_points[1, :]])
        #     else:
        #         clockwise_points.extend([right_points[1, :], right_points[0, :]])
        #
        #     clockwise_points = torch.cat(clockwise_points, dim=0)
        #     clockwise_bbox.append(clockwise_points)
        # clockwise_bbox = torch.stack(clockwise_bbox, dim=0).reshape((-1, 8)).to(self.bbox.dtype)
        # clockwise_bbox = BoxList(clockwise_bbox, image_size=self.size, mode='xyxyxyxy')
        # clockwise_bbox._copy_extra_fields(self)
        # #return clockwise_bbox

        _, right_points_index = self.bbox[:, 0:8:2].topk(k=2, dim=-1, largest=True)
        left_points_index = 3 - right_points_index
        index_temp = ((right_points_index - torch.tensor([1, 2]).to(self.device)).sum(dim=-1) == 0) | \
                     ((right_points_index - torch.tensor([2, 1]).to(self.device)).sum(dim=-1) == 0)
        left_points_index[index_temp] = torch.tensor([0, 3]).to(self.device)

        index_temp = (((right_points_index - torch.tensor([0, 3]).to(self.device)).sum(dim=-1) == 0) | \
                     ((right_points_index - torch.tensor([3, 0]).to(self.device)).sum(dim=-1) == 0)) & (~ index_temp)
        left_points_index[index_temp] = torch.tensor(([1, 2])).to(self.device)
        left_points1 = torch.cat([torch.gather(self.bbox, 1, (left_points_index[:, 0]*2).view(-1, 1)),
                                torch.gather(self.bbox, 1, (left_points_index[:, 0]*2+1).view(-1, 1))],
                                dim=-1)
        left_points2 = torch.cat([torch.gather(self.bbox, 1, (left_points_index[:, 1]*2).view(-1, 1)),
                                torch.gather(self.bbox, 1, (left_points_index[:, 1]*2+1).view(-1, 1))],
                                dim=-1)
        left_points = torch.cat([left_points1, left_points2], dim=-1)
        _, left_upper_points_index = left_points[:, 1:4:2].topk(k=2, dim=-1, largest=False)
        left_upper_points = torch.cat([torch.gather(left_points, 1, (left_upper_points_index[:, 0]*2).view(-1, 1)),
                                       torch.gather(left_points, 1, (left_upper_points_index[:, 0]*2+1).view(-1, 1))],
                                      dim=-1)
        left_bottom_points = torch.cat([torch.gather(left_points, 1, (left_upper_points_index[:, 1] * 2).view(-1, 1)),
                                       torch.gather(left_points, 1, (left_upper_points_index[:, 1] * 2 + 1).view(-1, 1))],
                                       dim=-1)

        right_points1 = torch.cat([torch.gather(self.bbox, 1, (right_points_index[:, 0] * 2).view(-1, 1)),
                                  torch.gather(self.bbox, 1, (right_points_index[:, 0] * 2 + 1).view(-1, 1))],
                                 dim=-1)
        right_points2 = torch.cat([torch.gather(self.bbox, 1, (right_points_index[:, 1] * 2).view(-1, 1)),
                                  torch.gather(self.bbox, 1, (right_points_index[:, 1] * 2 + 1).view(-1, 1))],
                                 dim=-1)
        right_points = torch.cat([right_points1, right_points2], dim=-1)
        _, right_upper_points_index = right_points[:, 1:4:2].topk(k=2, dim=-1, largest=False)
        right_upper_points = torch.cat([torch.gather(right_points, 1,(right_upper_points_index[:, 0] * 2).view(-1, 1)),
                                       torch.gather(right_points, 1, (right_upper_points_index[:, 0] * 2 + 1).view(-1, 1))],
                                      dim=-1)
        right_bottom_points = torch.cat([torch.gather(right_points, 1, (right_upper_points_index[:, 1] * 2).view(-1, 1)),
                                        torch.gather(right_points, 1, (right_upper_points_index[:, 1] * 2 + 1).view(-1, 1))],
                                       dim=-1)

        clock_wise_points = torch.cat([left_upper_points, left_bottom_points, right_bottom_points, right_upper_points],
                                        dim=-1)
        clockwise_box = BoxList(clock_wise_points.reshape(-1, 8), image_size=self.size, mode='xyxyxyxy')
        clockwise_box._copy_extra_fields(self)
        # if (clockwise_box.bbox - clockwise_bbox.bbox).sum() > 0:
        #     breakpoint =1
        return clockwise_box






    def _split_into_xyxyxyxy(self):
        if self.mode == "xyxyxyxy":

            x1, y1, x2, y2, x3, y3, x4, y4 = self.change_order_to_clockwise().bbox.split(1, dim=-1)
            return x1, y1, x2, y2, x3, y3, x4, y4
        elif self.mode == "xywha":
            xyxyxyxy_bbox = self.convert('xyxyxyxy')
            x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy_bbox.bbox.split(1, dim=-1)

            return x1, y1, x2, y2, x3, y3, x4, y4
        elif self.mode == 'xywha_d':
            xywha_bbox = self.convert('xywha')
            xyxyxyxy_bbox = xywha_bbox.convert('xyxyxyxy')
            x1, y1, x2, y2, x3, y3, x4, y4 = xyxyxyxy_bbox.bbox.split(1, dim=-1)

            return x1, y1, x2, y2, x3, y3, x4, y4
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            bbox = self._split_into_xyxyxyxy()
            bbox = torch.cat(bbox,dim=-1)
            scaled_box = bbox * ratio
            bbox = BoxList(scaled_box, size, mode='xyxyxyxy')
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox.convert(self.mode)

        ratio_width, ratio_height = ratios
        x1, y1, x2, y2, x3, y3, x4, y4 = self._split_into_xyxyxyxy()
        scaled_x1 = x1 * ratio_width
        scaled_x2 = x2 * ratio_width
        scaled_x3 = x3 * ratio_width
        scaled_x4 = x4 * ratio_width
        scaled_y1 = y1 * ratio_height
        scaled_y2 = y2 * ratio_height
        scaled_y3 = y3 * ratio_height
        scaled_y4 = y4 * ratio_height
        scaled_box = torch.cat(
            (scaled_x1, scaled_y1, scaled_x2, scaled_y2, scaled_x3, scaled_y3, scaled_x4, scaled_y4), dim=-1
        )
        bbox = BoxList(scaled_box, size, mode="xyxyxyxy")
        bbox = bbox.change_order_to_clockwise()
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (vertical flip or Horizontal flip)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        if self.mode == 'xywha_d':
            xc, yc, w, h, a = self.bbox.split(1,dim=1)
            if method == FLIP_LEFT_RIGHT:
                TO_REMOVE = 1
                transposed_xc = image_width - xc - TO_REMOVE
                transposed_yc = yc
                transposed_w = h
                transposed_h = w
                transposed_a = -90 - a
            if method == FLIP_TOP_BOTTOM:
                transposed_xc = xc
                transposed_yc = image_height - yc
                transposed_w = h
                transposed_h = w
                transposed_a = -90 - a
            transposed_boxes = torch.cat(
                (transposed_xc, transposed_yc, transposed_w, transposed_h, transposed_a), dim=-1
            )
            transposed_boxes = BoxList(transposed_boxes, image_size=self.size, mode='xywha_d')
        elif self.mode == 'xywha':
            xc, yc, w, h, a = self.bbox.split(1,dim=1)
            if method == FLIP_LEFT_RIGHT:
                TO_REMOVE = 1
                transposed_xc = image_width - xc - TO_REMOVE
                transposed_yc = yc
                transposed_w = h
                transposed_h = w
                transposed_a = -math.pi/2 - a
            if method == FLIP_TOP_BOTTOM:
                transposed_xc = xc
                transposed_yc = image_height - yc
                transposed_w = h
                transposed_h = w
                transposed_a = -math.pi/2 - a
            transposed_boxes = torch.cat(
                (transposed_xc, transposed_yc, transposed_w, transposed_h,transposed_a), dim=-1
            )
            transposed_boxes = BoxList(transposed_boxes, self.size, mode="xywha")
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = self.bbox.split(1,dim=1)
            if method == FLIP_LEFT_RIGHT:
                TO_REMOVE = 1
                transposed_x1 = image_width - x1 - TO_REMOVE
                transposed_x2 = image_width - x2 - TO_REMOVE
                transposed_x3 = image_width - x3 - TO_REMOVE
                transposed_x4 = image_width - x4 - TO_REMOVE
                transposed_y1 = y1
                transposed_y2 = y2
                transposed_y3 = y3
                transposed_y4 = y4
            elif method == FLIP_TOP_BOTTOM:
                transposed_x1 = x1
                transposed_x2 = x2
                transposed_x3 = x3
                transposed_x4 = x4
                transposed_y1 = image_height - y1
                transposed_y2 = image_height - y2
                transposed_y3 = image_height - y3
                transposed_y4 = image_height - y4

            transposed_boxes = torch.cat(
                (transposed_x1, transposed_y1, transposed_x2, transposed_y2, transposed_x3, transposed_y3,
                 transposed_x4, transposed_y4), dim=-1
            )
            transposed_boxes = BoxList(transposed_boxes, self.size, mode="xyxyxyxy")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            transposed_boxes.add_field(k, v)
        return transposed_boxes


    def rotate_90(self):
        '''
        only support anticlockwise rotation
        '''
        filpup_bbox = self.transpose(FLIP_LEFT_RIGHT)
        xyxyxyxy_bbox = filpup_bbox.convert('xyxyxyxy')
        bbox = xyxyxyxy_bbox.bbox.new_zeros(size=xyxyxyxy_bbox.bbox.shape)
        bbox[:, 0:8:2] = xyxyxyxy_bbox.bbox[:, 1:8:2]
        bbox[:, 1:8:2] = xyxyxyxy_bbox.bbox[:, 0:8:2]
        rotated_bbox = BoxList(bbox, image_size=(self.size[1], self.size[0],), mode='xyxyxyxy')
        rotated_bbox = rotated_bbox.change_order_to_clockwise()
        rotated_bbox = rotated_bbox.convert(self.mode)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.rotate_90()
            rotated_bbox.add_field(k, v)
        return rotated_bbox


    def crop(self, box,filter_bbox=False):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = self._split_into_xyxyxyxy()
        x1_, y1_, x2_, y2_, x3_, y3_, x4_, y4_ = x1.clone(), y1.clone(), x2.clone(), y2.clone(),\
                                                 x3.clone(), y3.clone(), x4.clone(), y4.clone()


        w, h = box[2] - box[0], box[3] - box[1]
        cropped_x1 = (x1_ - box[0]).clamp(min=0, max=w)
        cropped_y1 = (y1_ - box[1]).clamp(min=0, max=h)
        cropped_x2 = (x2_ - box[0]).clamp(min=0, max=w)
        cropped_y2 = (y2_ - box[1]).clamp(min=0, max=h)
        cropped_x3 = (x3_ - box[0]).clamp(min=0, max=w)
        cropped_y3 = (y3_ - box[1]).clamp(min=0, max=h)
        cropped_x4 = (x4_ - box[0]).clamp(min=0, max=w)
        cropped_y4 = (y4_ - box[1]).clamp(min=0, max=h)


        keep = ((cropped_y1 <= cropped_y2) & (cropped_x3 >= cropped_x2)
                & (cropped_x4 > cropped_x1) & (cropped_y3 > cropped_y4))


        crashed_bbox = (cropped_x1 == 0) & (cropped_x4 > cropped_x1) & keep
        cropped_y1[crashed_bbox] = y1[crashed_bbox] - box[1] + (y4[crashed_bbox] - y1[crashed_bbox]) \
                                   / (x4[crashed_bbox] - x1[crashed_bbox]) * (box[0] - x1[crashed_bbox])


        crashed_bbox = (cropped_x2 == 0) & (cropped_x3 > cropped_x2) & keep
        cropped_y2[crashed_bbox] = y2[crashed_bbox] - box[1]+ (y3[crashed_bbox] - y2[crashed_bbox]) / \
                                   (x3[crashed_bbox] - x2[crashed_bbox]) * (box[0] - x2[crashed_bbox])


        crashed_bbox = (cropped_x4 == w) & (cropped_x1 < cropped_x4) & keep
        cropped_y4[crashed_bbox] = y1[crashed_bbox] - box[1]+ (y4[crashed_bbox] - y1[crashed_bbox]) /\
                                   (x4[crashed_bbox] - x1[crashed_bbox]) * (box[2] - x1[crashed_bbox])


        crashed_bbox = (cropped_x3 == w) & (cropped_x2 < cropped_x3) & keep
        cropped_y3[crashed_bbox] = y2[crashed_bbox] - box[1] + (y3[crashed_bbox] - y2[crashed_bbox]) / \
                     (x3[crashed_bbox] - x2[crashed_bbox]) * (box[2] - x2[crashed_bbox])


        crashed_bbox = (cropped_y1 == 0) & (cropped_y2 > cropped_y1) & keep
        cropped_x1[crashed_bbox] = x1[crashed_bbox] - box[0] + (x2[crashed_bbox] - x1[crashed_bbox]) / \
                                   (y2[crashed_bbox] - y1[crashed_bbox]) * (box[1] - y1[crashed_bbox])


        crashed_bbox = (cropped_y4 == 0) & (cropped_y3 > cropped_y4) & keep
        cropped_x4[crashed_bbox] = x4[crashed_bbox]-box[0] + (x3[crashed_bbox] - x4[crashed_bbox]) / \
                                   (y3[crashed_bbox] - y4[crashed_bbox]) * (box[1] - y4[crashed_bbox])


        crashed_bbox = (cropped_y2 == h) & (cropped_y1 < cropped_y2) & keep
        cropped_x2[crashed_bbox] = x1[crashed_bbox] -box[0] + (x2[crashed_bbox] - x1[crashed_bbox]) / \
                                   (y2[crashed_bbox] - y1[crashed_bbox]) * (box[3] - y1[crashed_bbox])


        crashed_bbox = (cropped_y3 == h) & (cropped_y4 < cropped_y3) & keep
        cropped_x3[crashed_bbox] = x4[crashed_bbox] -box[0] + (x3[crashed_bbox] - x4[crashed_bbox]) / \
                                   (y3[crashed_bbox] - y4[crashed_bbox]) * (box[3] - y4[crashed_bbox])




        cropped_bboxes = torch.cat(
            (cropped_x1, cropped_y1, cropped_x2, cropped_y2,
             cropped_x3, cropped_y3, cropped_x4, cropped_y4), dim=-1
        )

        bbox = BoxList(cropped_bboxes, (w, h), mode="xyxyxyxy")
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)




    # Tensor-like methods
    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        x1, y1, x2, y2, x3, y3, x4, y4 = self._split_into_xyxyxyxy()
        x1 = x1.clamp_(min=0, max=self.size[0] - TO_REMOVE).squeeze()
        x2 = x2.clamp_(min=0, max=self.size[0] - TO_REMOVE).squeeze()
        x3 = x3.clamp_(min=0, max=self.size[0] - TO_REMOVE).squeeze()
        x4 = x4.clamp_(min=0, max=self.size[0] - TO_REMOVE).squeeze()
        y1 = y1.clamp_(min=0, max=self.size[1] - TO_REMOVE).squeeze()
        y2 = y2.clamp_(min=0, max=self.size[1] - TO_REMOVE).squeeze()
        y3 = y3.clamp_(min=0, max=self.size[1] - TO_REMOVE).squeeze()
        y4 = y4.clamp_(min=0, max=self.size[1] - TO_REMOVE).squeeze()

        bbox = BoxList(torch.stack([x1, y1, x2, y2, x3, y3, x4, y4],dim=-1).reshape(-1,8),image_size=self.size,mode='xyxyxyxy')
        bbox._copy_extra_fields(self)
        bbox = bbox.convert(self.mode)

        if remove_empty:
            # keep = ((box[:, 3]- box[:,1]) > 2) & ((box[:, 2] - box[:,0])> 2)
            # p1_y < p2_y p3_x > p2_x p4_x > p1_x p3_y > p4_y
            # if a = -pi/2

            keep = (((y1 <= y2) & (x3 >= x2) & (x4 >= x1) & (y3 >= y4)) & \
                   ~((y1==y2) & (x3==x2) & (x4==x1) & (y3==y4))).reshape(-1)

            return bbox[keep]
        return bbox




    def area(self):
        box = self.bbox
        if self.mode == "xyxyxyxy":
            # TODO here is a imprecise calculation, assuming the bbox is a rectangle
            xywha_bbox = self.convert('xywha')
            area = xywha_bbox.bbox[:,2] * xywha_bbox.bbox[:,3]

        elif self.mode == "xywha":
            area = box[:, 2] * box[:, 3]
        elif self.mode == "xywha_d":
            area = box[:, 2] * box[:, 3]
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


    def xyxyxyxy_to_xywha(self):
        '''
        convert xyxyxyxy format to xywha
        '''
        assert self.mode == 'xyxyxyxy'
        rbboxes = []
        indexs = []
        bboxes = self.bbox.numpy()
        for i, bbox in enumerate(bboxes):
            rbbox = cv2.minAreaRect(bbox.reshape((4,2)).astype(np.float32))
            x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]

            while not 0 > a >= -90:
                if a >= 0:
                    a -= 90
                    w, h = h, w
                else:
                    a += 90
                    w, h = h, w
            a = a / 180 * np.pi
            # - pi/ 2 <= a <= 0
            assert 0 > a >= -np.pi / 2

            rbboxes.append([x,y,w,h,a])
            indexs.append(i)


        rbbox = torch.tensor(rbboxes,dtype=torch.float32,device=self.device)

        rbbox = BoxList(rbbox,image_size=self.size,mode='xywha')

        for k, v in self.extra_fields.items():
            rbbox.add_field(k, v[indexs])
        return rbbox



    def xywha_to_xyxyxyxy(self, clockwise=True):
        """Convert xywha format to xyxyxyxyxy

        xyxyxyxy : left_top point xy, left_bottom point xy, right_bottom point xy, right top point xy
        """
        assert self.mode == 'xywha'
        xc = self.bbox[:, 0]
        yc = self.bbox[:, 1]
        w = self.bbox[:, 2]
        h = self.bbox[:, 3]
        a = self.bbox[:, 4]
        # -pi/2 <= a < 0, cosa>0 sina<0
        cosa = torch.cos(a)
        sina = torch.sin(a)
        # wx>0 wy<0
        wx, wy = w / 2 * cosa, w / 2 * sina
        # hx>0 hy>0
        hx, hy = -h / 2 * sina, h / 2 * cosa
        p1x, p1y = xc - wx - hx, yc - wy - hy
        p2x, p2y = xc - wx + hx, yc - wy + hy
        p3x, p3y = xc + wx + hx, yc + wy + hy
        p4x, p4y = xc + wx - hx, yc + wy - hy


        polygons = torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1).to(self.device)
        rbbox = BoxList(polygons, image_size=self.size, mode='xyxyxyxy')
        rbbox._copy_extra_fields(self)
        if clockwise:
            rbbox = rbbox.change_order_to_clockwise()
        return rbbox

    def xywha_to_xywhad(self):
        '''
        Convert xywha format to xywha_d
        for 'xywha' the angle means radian (0 < angel <= -pi/2)
        for 'xywha_d' the angle means degress (0 < angel_d <= -90)
        '''
        assert self.mode == 'xywha'
        a = self.bbox[:, 4]
        degree_a = a /math.pi*180
        bbox = self.bbox.new_zeros(self.bbox.shape)
        bbox[:,4] = degree_a
        bbox[:,:4] = self.bbox[:,:4]
        degreea_bbox = BoxList(bbox,image_size=self.size,mode='xywha_d')
        degreea_bbox._copy_extra_fields(self)
        return degreea_bbox

    def xywhad_to_xywha(self):
        '''
        Convert xywha_d format to xywha
        '''
        assert self.mode == 'xywha_d'
        a = self.bbox[:, 4]
        radian_a = a/180 * math.pi
        bbox = self.bbox.new_zeros(self.bbox.shape)
        bbox[:, 4] = radian_a
        bbox[:, :4] = self.bbox[:, :4]
        radian_bbox = BoxList(bbox, image_size=self.size, mode='xywha')
        radian_bbox._copy_extra_fields(self)
        return radian_bbox

    def xywhad_to_xyxyxyxy(self):
        '''
        convert xywha_d foramt to xyxyxyxy
        '''
        assert self.mode == 'xywha_d'
        xywha_bbox = self.xywhad_to_xywha()
        xyxyxyxy_bbox = xywha_bbox.xywha_to_xyxyxyxy()
        return xyxyxyxy_bbox

    def xyxyxyxy_to_xywhad(self):
        '''
        convert xyxyxyxy format to xywha_d
        '''
        assert self.mode == 'xyxyxyxy'
        xywha_bbox = self.xyxyxyxy_to_xywha()
        xywhad_bbox = xywha_bbox.xywha_to_xywhad()
        return xywhad_bbox

    def xywhad_round(self):
        '''
        round angle(degree) to integral
        '''

        assert self.mode == 'xywha_d'
        rounded_xywhad = self.bbox.new_zeros(self.bbox.shape)
        rounded_angle = torch.round(self.bbox[:, 4])
        origin_bbox = self.bbox[:, :4].clone()
        index_for_0 = (rounded_angle == 0)
        rounded_angle[index_for_0] = -90
        origin_bbox[index_for_0, 2], origin_bbox[index_for_0, 3] = origin_bbox[index_for_0, 3], origin_bbox[index_for_0, 2]

        assert torch.max(rounded_angle) < 0, "degree angle < 0"
        rounded_xywhad[:, :4] = origin_bbox[:, :4]
        rounded_xywhad[:, 4] = rounded_angle
        rounded_xywhad = BoxList(rounded_xywhad, image_size=self.size, mode='xywha_d')
        rounded_xywhad._copy_extra_fields(self)
        return rounded_xywhad





def filter_bboxes(origin_bboxes, cropped_bboxes,bbox,keep_best_ioa=False):


    if cropped_bboxes.mode == 'xywha' or cropped_bboxes.mode == 'xywha_d':
        keep = (cropped_bboxes.bbox[:, 2] > 0) & (cropped_bboxes.bbox[:, 3] > 0)
        cropped_bboxes_ = cropped_bboxes[keep]
        cropped_bboxes_filtered = copy.copy(cropped_bboxes_)
    else:
        x1, y1, x2, y2, x3, y3, x4, y4 = cropped_bboxes._split_into_xyxyxyxy()
        keep = ((y1 <= y2) & (x3 >= x2) & (x4 > x1) & (y3 > y4)).reshape(-1)
        cropped_bboxes_ = cropped_bboxes[keep]
        cropped_bboxes_filtered = copy.copy(cropped_bboxes_)


    origin_bboxes_ = origin_bboxes.convert('xyxyxyxy')
    origin_bboxes_ = origin_bboxes_[keep].bbox

    cropped_bboxes_ = cropped_bboxes_.convert('xyxyxyxy')

    cropped_bboxes_ = cropped_bboxes_.bbox.clone()
    cropped_bboxes_[:, 0:8:2] += bbox[0]
    cropped_bboxes_[:, 1:8:2] += bbox[1]

    ioas = cropped_bboxes_.new_zeros(cropped_bboxes_.shape[0])

    for i, (cropped_bbox, origin_bbox) in enumerate(zip(cropped_bboxes_, origin_bboxes_)):
        if (cropped_bbox[1] <= cropped_bbox[3]) and (cropped_bbox[4] >= cropped_bbox[2]) and \
            (cropped_bbox[6] > cropped_bbox[0]) and (cropped_bbox[5] > cropped_bbox[7]):
            boxlist1 = BoxList(cropped_bbox.reshape(-1, 8), image_size=cropped_bboxes.size, mode='xyxyxyxy')
            boxlist2 = BoxList(origin_bbox.reshape(-1, 8), image_size=cropped_bboxes.size, mode='xyxyxyxy')
            ioa = boxlist_ioa(boxlist1, boxlist2)
            ioas[i] = ioa

    keep = ioas > 0.6
    if keep_best_ioa:
        _, ioa_max_index = torch.max(ioas, dim=0)
        keep[ioa_max_index] = True
    return cropped_bboxes_filtered[keep].convert(cropped_bboxes.mode)



def boxlist_rnms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xywha")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = rnms(
        torch.cat([boxes, score[:, None]], -1), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)



def boxlist_ml_rnms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xywha")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)

    keep = batched_rnms(boxes, scores, labels.to(torch.float32), nms_thresh, class_agnostic=True)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    area = boxlist.area()
    keep = area > min_size
    return boxlist[keep]


def polygon_overlaps(polygons1, polygons2):
    p1 = torch.tensor(polygons1[:, :8], dtype=torch.float64)  # in case the last element of a row is the probability
    p2 = torch.tensor(polygons2[:, :8], dtype=torch.float64)  # in case the last element of a row is the probability
    return polygon_iou(p1, p2).numpy()

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,5] or [N,8].
            if bounding boxes are [N,5], then it will be converted to xyxyxyxy
      box2: (BoxList) bounding boxes, sized [M,5] or [M,8].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    boxlist1_ = copy.deepcopy(boxlist1)
    boxlist2_ = copy.deepcopy(boxlist2)

    # actually , there is no need to change_order_to_clockwise
    if boxlist1_.mode == 'xywha_d':
        boxlist1_ = boxlist1_.xywhad_to_xywha().xywha_to_xyxyxyxy(clockwise=False)
    elif boxlist1_.mode == 'xywha':
        boxlist1_ = boxlist1_.xywha_to_xyxyxyxy(clockwise=False)
    else:
        pass
    if boxlist2_.mode == 'xywha_d':
        boxlist2_ = boxlist2_.xywhad_to_xywha().xywha_to_xyxyxyxy(clockwise=False)
    elif boxlist2_.mode == 'xywha':
        boxlist2_ = boxlist2_.xywha_to_xyxyxyxy(clockwise=False)
    else:
        pass

    # boxlist1_ = boxlist1_.convert("xyxyxyxy")
    # boxlist2_ = boxlist2_.convert("xyxyxyxy")

    p1 = boxlist1_.bbox.to(torch.float64).cpu()
    p2 = boxlist2_.bbox.to(torch.float64).cpu()
    iou = polygon_iou(p1, p2).numpy()

    return torch.from_numpy(iou).to(boxlist1.device)

def boxlist_ioa(boxlist1, boxlist2):
    """Returns the intersection over box2 area given box1, box2.
    Notice: not fast implemented
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))


    area2 = boxlist2.area()

    boxlist1_ = boxlist1.convert('xywha_d')
    boxlist2_ = boxlist2.convert('xywha_d')
    box1, box2 = boxlist1_.bbox, boxlist2_.bbox

    inter = box1.new_zeros((len(boxlist1_), len(boxlist2_)))
    for i, _ in enumerate(box1):
        for j, _ in enumerate(box2):
            box1_ = copy.deepcopy(box1[i, :]).numpy()
            box1_ = ((box1_[0], box1_[1]), (box1_[2], box1_[3]), box1_[4])
            box2_ = copy.deepcopy(box2[j, :]).numpy()
            box2_ = ((box2_[0], box2_[1]), (box2_[2], box2_[3]), box2_[4])
            inter_area = cv2.rotatedRectangleIntersection(box1_, box2_)
            if inter_area[0] == 0:
                inter_area = 0
            else:
                order_pts = cv2.convexHull(inter_area[1], returnPoints=True)
                inter_area = cv2.contourArea(order_pts)
            inter[i, j] = inter_area
    ioa = inter / area2
    return ioa

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def test():
    box1 = torch.tensor([1, 2, 3, 4, 1,4, 3,2]).repeat(4,1).to(torch.float64)
    box2 = torch.tensor([1.5, 2, 1.5,5, 4,5, 4,2]).repeat(4,1).to(torch.float64)
    box1 = BoxList(box1, image_size=(1024,1024),mode='xyxyxyxy')
    box2 = BoxList(box2, image_size=(1024,1024),mode='xyxyxyxy')
    ious = boxlist_iou(box1, box2)
    print(ious)