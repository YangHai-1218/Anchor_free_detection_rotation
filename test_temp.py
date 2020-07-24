import torch
import torch.nn as nn
from backbone.Efficientnet import EfficientnetWithBiFPN



model = EfficientnetWithBiFPN(compound_coef=5,load_total_weight=True,
                                 total_weight_path='../Yet-Another-EfficientDet-Pytorch/weights/efficientdet-d5.pth')
breakpoint = 1
