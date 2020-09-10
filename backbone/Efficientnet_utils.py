import torch
from torch import nn
from torch.nn import functional as F
import math
from utils.base_conv import Conv2dStaticSamePadding
from utils.activation import (
    MemoryEfficientSwish,
    Swish,
)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, effnet_param):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - effnet_param.batch_norm_momentum
        self._bn_eps = effnet_param.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d()

        # Expansion phase, stride = 1,
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase, downsampling
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        # Only when the input and output are of the same dimension, do skip connection
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()







def drop_connect(inputs, p, training):
    """ Drop connect. """
    """ what is the drop-connect: https: // github.com / zylo117 / Yet - Another - EfficientDet - Pytorch / issues / 271 """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output







def get_same_padding_conv2d():
    return Conv2dStaticSamePadding


class Efficnetnet_param():
    def __init__(self,compound_coef=0,):
        # global parameters setting
        self.compound_coef = compound_coef
        self.depth_divisor = 8
        self.min_depth = None
        self.batch_norm_momentum =0.99
        self.batch_norm_epsilon = 1e-3
        self.dropout_connect_rate = 0.2
        self.width_cofficient, self.depth_cofficient, self.resolution_cofficient, self.dropout_rate = self.decode()
        self.Efficient_b0()
        self.depth_divisor = 8
        self.min_depth = None

        self.stem_output_filters = self.stem_output_filters_cal()
        block1 = self.b0_block1.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        block2 = self.b0_block2.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        block3 = self.b0_block3.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        block4 = self.b0_block4.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        block5 = self.b0_block5.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        block6 = self.b0_block6.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        block7 = self.b0_block7.round_filers_repeats(width_cofficient=self.width_cofficient,depth_cofficient=self.depth_cofficient)
        self.blocks = [block1,block2,block3,block4,block5,block6,block7]

    def decode(self):
        params_dict = {
            # Coefficients:   width,depth,res,dropout
            'efficientnet-b0': (1.0, 1.0, 224, 0.2),
            'efficientnet-b1': (1.0, 1.1, 240, 0.2),
            'efficientnet-b2': (1.1, 1.2, 260, 0.3),
            'efficientnet-b3': (1.2, 1.4, 300, 0.3),
            'efficientnet-b4': (1.4, 1.8, 380, 0.4),
            'efficientnet-b5': (1.6, 2.2, 456, 0.4),
            'efficientnet-b6': (1.8, 2.6, 528, 0.5),
            'efficientnet-b7': (2.0, 3.1, 600, 0.5),
            'efficientnet-b8': (2.2, 3.6, 672, 0.5),
            'efficientnet-l2': (4.3, 5.3, 800, 0.5),
        }
        model_name = 'efficientnet-b'+str(self.compound_coef)
        return params_dict[model_name]


    def stem_output_filters_cal(self):
        multiplier = self.width_cofficient
        if not multiplier:
            return self.b0_stem_output_filters
        divisor = self.depth_divisor
        min_depth = self.min_depth
        filters = self.b0_stem_output_filters
        filters *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
            new_filters += divisor
        return int(new_filters)


    def Efficient_b0(self):
        # input (N,H,W,3) output (N,112,112,32)
        self.b0_stem_output_filters = 32
        # input (N,112,112,32) output(N,112,112,16)
        self.b0_block1 = Block_args(input_filters=32,output_filters=16,num_repeat=1,stride=1,kernel_size=3,expand_ratio=1)
        # input (N,112,112,16) output(N,56,56,24)
        self.b0_block2 = Block_args(input_filters=16,output_filters=24,num_repeat=2,stride=2,kernel_size=3,expand_ratio=6)
        # input (N,56,56,24) output(N, 28,28,40)
        self.b0_block3 = Block_args(input_filters=24,output_filters=40,num_repeat=2,stride=2,kernel_size=5,expand_ratio=6)
        # input (N,28,28,40) output(N, 14,14,80)
        self.b0_block4 = Block_args(input_filters=40,output_filters=80,num_repeat=3,stride=2,kernel_size=3,expand_ratio=6)
        # input (N, 14,14,80) output(N,14,14,112)
        self.b0_block5 = Block_args(input_filters=80,output_filters=112,num_repeat=3,stride=1,kernel_size=5,expand_ratio=6)
        # input (N, 14,14,112) output(N,7,7,192)
        self.b0_block6 = Block_args(input_filters=112,output_filters=192,num_repeat=4,stride=2,kernel_size=5,expand_ratio=6)
        # input (N, 7, 7, 192) output (N, 7, 7, 320)
        self.b0_block7 = Block_args(input_filters=192,output_filters=320,num_repeat=1,stride=1,kernel_size=3,expand_ratio=6)

    @staticmethod
    def get_model_params(model_name):
        compound_coef = int(model_name[14:])
        return Efficnetnet_param(compound_coef)

class Block_args():
    def __init__(self,input_filters,output_filters,num_repeat,stride,kernel_size=3,expand_ratio=6,se_ratio=0.25,id_skip=True):
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.id_skip = id_skip
        self.num_repeat = num_repeat
        self.stride = stride

    def round_filters(self, width_cofficient,depth_divisor=8,min_depth=None):
        multiplier = width_cofficient
        if not multiplier:
            return 0
        filters = [self.input_filters,self.output_filters]
        divisor = depth_divisor
        min_depth = min_depth
        new_filters = []
        for filter in filters:
            filter *= multiplier
            min_depth = min_depth or divisor
            new_filter = max(min_depth, int(filter + divisor / 2) // divisor * divisor)
            if new_filter < 0.9 * filter:  # prevent rounding by more than 10%
                new_filter += divisor
            new_filters.append(new_filter)
        new_input_filters = new_filters[0]
        new_output_filters = new_filters[1]
        return Block_args(input_filters=new_input_filters,output_filters=new_output_filters,num_repeat=self.num_repeat,stride=self.stride,
                          kernel_size=self.kernel_size,expand_ratio=self.expand_ratio)

    def round_repeats(self,depth_cofficient):

        if not depth_cofficient:
            return self
        new_num_repeat = int(math.ceil(self.num_repeat * depth_cofficient))
        return Block_args(input_filters=self.input_filters,output_filters=self.output_filters,num_repeat=new_num_repeat,
                          stride=self.stride,kernel_size=self.kernel_size,expand_ratio=self.expand_ratio)



    def round_filers_repeats(self,width_cofficient,depth_cofficient):
        rounded_filters_block = self.round_filters(width_cofficient)
        rounded_block = rounded_filters_block.round_repeats(depth_cofficient)
        return rounded_block




url_map_advprop = {
    'efficientnet-b0': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b0-b64d5a18.pth',
    'efficientnet-b1': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b1-0f3ce85a.pth',
    'efficientnet-b2': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b2-6e9d97e5.pth',
    'efficientnet-b3': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b3-cdd7c0f4.pth',
    'efficientnet-b4': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b4-44fb3a87.pth',
    'efficientnet-b5': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b5-86493f6b.pth',
    'efficientnet-b6': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b6-ac80338e.pth',
    'efficientnet-b7': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b7-4652b6dd.pth',
    'efficientnet-b8': 'https://publicmodels.blob.core.windows.net/container/advprop/efficientnet-b8-22a8fe65.pth',
}
def load_pretrained_weights(model, model_name, load_fc=True, advprop=False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map
    state_dict = model_zoo.load_url(url_map_[model_name], map_location=torch.device('cpu'))
    # state_dict = torch.load('../../weights/backbone_efficientnetb0.pth')
    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        print(ret)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))



def test():
    B0 = Efficnetnet_param(compound_coef=0)

    breakpoint = 1


if __name__ =="__main__":
    test()