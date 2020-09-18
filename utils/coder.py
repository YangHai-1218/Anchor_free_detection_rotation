import torch
import math

class BoxCoder(object):
    """Delta XYWHA BBox coder

    this coder is used for rotated objects detection (for example on task1 of DOTA dataset).
    this coder encodes bbox (xc, yc, w, h, a) into delta (dx, dy, dw, dh, da) and
    decodes delta (dx, dy, dw, dh, da) back to original bbox (xc, yc, w, h, a).

    Args:
        target_means (Sequence[float]): denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): denormalizing standard deviation of
            target for delta coordinates
        """
    def __init__(self, regression_type, anchor_sizes, anchor_strides):
        self.regression_type = regression_type
        self.anchor_sizes = anchor_sizes
        self.anchor_strides = anchor_strides

    def encode(self, gt_boxes, anchors):
        '''
        Args:
            gt_boxes : tensor shape N*5 xc,yc,w,h,angle(degree)
            anchors : tensor shape N*5, xc,yc,w,h,angle(degree)

        '''

        ex_widths = anchors[:, 2]
        ex_heights = anchors[:, 3]
        ex_ctr_x = anchors[:, 0]
        ex_ctr_y = anchors[:, 1]

        gt_widths = gt_boxes[:, 2]
        gt_heights = gt_boxes[:, 3]
        gt_ctr_x = gt_boxes[:, 0]
        gt_ctr_y = gt_boxes[:, 1]
        angle = gt_boxes[:, 4]

        #wx, wy, ww, wh = (10., 10., 5., 5.)
        wx, wy, ww, wh = (5., 5., 1., 1.)
        # Normalize
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, angle), dim=1)

        return targets

    def decode(self, preds, anchors):
        '''
        Args:
            preds tensor shape N*5 dx,dy,dw,dh,angle
            anchors tesor shape N*5 xc,xy,w,h,angle
        '''

        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2]
        heights = anchors[:, 3]
        ctr_x = anchors[:, 0]
        ctr_y = anchors[:, 1]

        #wx, wy, ww, wh = (10., 10., 5., 5.)
        wx, wy, ww, wh = (5., 5., 1., 1.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x
        pred_boxes[:, 1::4] = pred_ctr_y
        pred_boxes[:, 2::4] = pred_w
        pred_boxes[:, 3::4] = pred_h
        pred_boxes[:, 4] = preds[:, 4]
        return pred_boxes


class PointCoder(BoxCoder):
    def __init__(self,regression_type, anchor_sizes, anchor_strides):
        super(PointCoder,self).__init__(regression_type, anchor_sizes, anchor_strides)
        assert self.regression_type == 'POINT'

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

        w = self.anchor_sizes[0] / self.anchor_strides[0]
        l = w * (anchors_cx - gt_boxes[:, 0]) / anchors_w
        t = w * (anchors_cy - gt_boxes[:, 1]) / anchors_h
        r = w * (gt_boxes[:, 2] - anchors_cx) / anchors_w
        b = w * (gt_boxes[:, 3] - anchors_cy) / anchors_h
        targets = torch.stack([l, t, r, b], dim=1)
        return targets


    def decode(self, preds, anchors):
        TO_REMOVE = 1  # TODO remove
        anchors_w = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        anchors_h = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2

        w = self.anchor_sizes[0] / self.anchor_strides[0]
        x1 = anchors_cx - preds[:, 0] / w * anchors_w
        y1 = anchors_cy - preds[:, 1] / w * anchors_h
        x2 = anchors_cx + preds[:, 2] / w * anchors_w
        y2 = anchors_cy + preds[:, 3] / w * anchors_h
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        return pred_boxes




def test():
    gt = torch.tensor([[10, 20, 3, 5, -30]]).to(torch.float32)
    anchor = torch.tensor([[10, 20, 32, 32, -90]]).to(torch.float32)
    coder = BoxCoder(regression_type='box',anchor_sizes=None,anchor_strides=None)
    reg_targets = coder.encode(gt,anchor)
    pred = coder.decode(reg_targets,anchor)
    print(reg_targets)
    print(pred)

if __name__ == '__main__':
    test()