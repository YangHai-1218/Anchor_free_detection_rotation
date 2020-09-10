import  torch
from torch.nn import init
from torch import nn
import math


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return init._no_grad_normal_(tensor, 0., std)


def focal_loss_init(tensor,prior):
    prior_prob = prior
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(tensor, bias_value)