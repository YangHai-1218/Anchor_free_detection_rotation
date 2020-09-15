import torch
from .boxlist import BoxList
from torch import nn
import numpy as np

class AnchorGenerator(nn.Module):
    """
    For a set of image sizes and feature maps, computes a set of anchors
    anchor size(w*h) ≈ sizes[i]**2 , eg:
        size = 256, aspect_ratios = (128,256,512)
        anchor('xyxy' mode)
            [[-175.5000,  -87.5000,  191.5000,  103.5000], --> aspect_raio = 0.5 size = 367*192 ≈ 256*256
            [-119.5000, -119.5000,  135.5000,  135.5000], --> aspect_ratio = 1, size =  255*255 ≈ 256*256
            [ -79.5000, -167.5000,   95.5000,  183.5000]] --> aspect_ratio = 2, size = 175*351 ≈ 256*256
    for now only support angel = 0
    """

    def __init__(
        self,
        sizes=(128, 256, 512),
        aspect_ratios=(0.5, 1.0, 2.0),
        anchor_strides=(8, 16, 32),
        angle=(-90),
        straddle_thresh=0,
    ):
        super(AnchorGenerator, self).__init__()

        if len(anchor_strides) == 1:
            anchor_stride = anchor_strides[0]
            cell_anchors = [
                generate_anchors(anchor_stride, sizes, aspect_ratios).float()
            ]
        else:
            if len(anchor_strides) != len(sizes):
                raise RuntimeError("FPN should have #anchor_strides == #sizes")

            #
            cell_anchors = [
                generate_anchors(
                    anchor_stride,
                    size if isinstance(size, (tuple, list)) else (size,),
                    aspect_ratios
                ).float()
                for anchor_stride, size in zip(anchor_strides, sizes)
            ]
        self.strides = anchor_strides
        self.cell_anchors = BufferList(cell_anchors)
        self.straddle_thresh = straddle_thresh
        self.angle = angle

    def num_anchors_per_location(self):
        return [len(cell_anchors) for cell_anchors in self.cell_anchors]

    def grid_anchors(self, grid_sizes):
        anchors = []
        for size, stride, base_anchors in zip(
            grid_sizes, self.strides, self.cell_anchors
        ):
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(
                0, grid_width * stride, step=stride, dtype=torch.float32, device=device
            )
            shifts_y = torch.arange(
                0, grid_height * stride, step=stride, dtype=torch.float32, device=device
            )
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors

    def add_visibility_to(self, boxlist):
        image_width, image_height = boxlist.size
        anchors = boxlist.bbox
        if self.straddle_thresh >= 0:
            inds_inside = (
                (anchors[..., 0] >= -self.straddle_thresh)
                & (anchors[..., 1] >= -self.straddle_thresh)
                & (anchors[..., 2] < image_width + self.straddle_thresh)
                & (anchors[..., 3] < image_height + self.straddle_thresh)
            )
        else:
            device = anchors.device
            inds_inside = torch.ones(anchors.shape[0], dtype=torch.uint8, device=device)
        boxlist.add_field("visibility", inds_inside)

    def forward(self, image_list, feature_maps):
        '''
        output: 每张featuremap上的anchor组为boxlist object, 单张image的不同feturemap anchor box (boxlist）组成list
                多张image 组成 list
        '''
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes)
        anchors = []
        for i, (image_height, image_width) in enumerate(image_list.sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_per_feature_map_with_angle = xyxy_to_xywha(anchors_per_feature_map)
                boxlist = BoxList(
                    anchors_per_feature_map_with_angle, (image_width, image_height), mode="xywha_d"
                )
                self.add_visibility_to(boxlist)
                anchors_in_image.append(boxlist)
            anchors.append(anchors_in_image)
        return anchors



def xyxy_to_xywha(anchors, angle=(-90)):
    xc = ((anchors[:, 0] + anchors[:, 2])/2).squeeze()
    yc = ((anchors[:, 1] + anchors[:, 3])/2).squeeze()
    w = ((anchors[:, 2] - anchors[:, 0])).squeeze()
    h = ((anchors[:, 3] - anchors[:, 1])).squeeze()

    assert anchors.shape[-1] == 4
    if isinstance(angle, (tuple, list)):
        angle = angle[0]
    anchors_with_angle = anchors.new_full((anchors.shape[-2], 5), angle)
    # Attention here, if the angle == -90, then w=h, h=w
    anchors_with_angle[:, :4] = torch.stack([xc, yc, h, w], dim=-1)

    return anchors_with_angle


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float),
    )

def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 0.5
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return torch.from_numpy(anchors)

def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        )
    )
    return anchors

def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

def test():
    #generator = AnchorGenerator()
    a = torch.zeros((20,4))
    xywha = xyxy_to_xywha(a)
    print(xywha)
    breakpoint = 1


