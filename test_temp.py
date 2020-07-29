import torch
import torch.nn as nn
from backbone.Efficientnet import EfficientnetWithBiFPN
from argument import get_args
from model import Efficientnet_Bifpn_ATSS

args = get_args()
model = Efficientnet_Bifpn_ATSS(args,0,load_backboe_weight=False)
