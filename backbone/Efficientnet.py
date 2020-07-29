import torch
from torch import nn
from torch.nn import functional as F
from .Efficientnet_utils import (
    MBConvBlock,
    get_same_padding_conv2d,
    Swish,
    Efficnetnet_param,
)
from utils.Activation import (
    MemoryEfficientSwish,
)
from FPN.BiFPN import BiFPN
import os
import copy



class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, effnet_param=None):
        super().__init__()

        self.param = effnet_param


        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d()

        # Batch norm parameters
        bn_mom = 1 - self.param.batch_norm_momentum
        bn_eps = self.param.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = self.param.stem_output_filters  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self.param.blocks:

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self.param))
            # other blocks only have to keep channel and keep size
            if block_args.num_repeat > 1:
                block_args_repeat = copy.deepcopy(block_args)
                block_args_repeat.input_filters = block_args.output_filters
                block_args_repeat.stride = 1
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args_repeat,self.param))

        self._swish = MemoryEfficientSwish()
    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)



    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self._conv_stem(inputs)
        x = self._bn0(x)
        x = self._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.param.dropout_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            # feture map extractor: after the block of downsampling , extract feature map
            if block._depthwise_conv.stride == [2, 2] or block._depthwise_conv.stride == (2,2):
                feature_maps.append(last_x)
            elif idx == len(self._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x

        return feature_maps[1:]

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        effnet_param = Efficnetnet_param.get_model_params(model_name)
        return cls(effnet_param=effnet_param)

    @classmethod
    def from_pretrained(cls, model_name, load_weights=True, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        # if load_weights:
        #     load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        # if in_channels != 3:
        #     Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
        #     out_channels = round_filters(32, model._global_params)
        #     model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    # @classmethod
    # def get_image_size(cls, model_name):
    #     cls._check_model_name_is_valid(model_name)
    #     _, _, res, _ = efficientnet_params(model_name)
    #     return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))



class EfficientnetWithBiFPN(nn.Module):
    def __init__(self,compound_coef,load_total_weight=False,total_weight_path=None):
        super(EfficientnetWithBiFPN, self).__init__()
        self.compound_coef = compound_coef
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5]
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
        }

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.backbone_net = EfficientNet.from_pretrained(f'efficientnet-b{compound_coef}')
        if load_total_weight and total_weight_path:
            self.load_total_weight(weight_path=total_weight_path)

    def forward(self,inputs):
        _, p3, p4, p5 = self.backbone_net(inputs)
        if False:
            print(f'p3 shape: {p3.shape}')
            print(f'p4 shape:{p4.shape}')
            print(f'p5 shape:{p5.shape}')

        features = (p3,p4,p5)
        (p3,p4,p5,p6,p7) = self.bifpn(features)
        if False:
            print(f'p3 shape: {p3.shape}')
            print(f'p4 shape:{p4.shape}')
            print(f'p5 shape:{p5.shape}')
            print(f'p6 shape: {p6.shape}')
            print(f'p7 shape:{p7.shape}')
        return (p3,p4,p5,p6,p7)

    def load_total_weight(self, weight_path):
        weights = torch.load(weight_path)
        weights_new = weights.copy()
        for k in weights:
            if k.startswith('regressor') or k.startswith('classifier'):
                del weights_new[k]
            if k.startswith('backbone_net'):
                v = weights[k]
                weights_new.pop(k)
                k_new = k[0:13] + k[19:]
                weights_new[k_new] = v
        self.load_state_dict(weights_new)
        weight_name = os.path.basename(weight_path)
        print(f'[INFO] load pretrained weights {weight_name} successfully!')




def test():
    effnet_with_bifpn = EfficientnetWithBiFPN(compound_coef=0,load_total_weight=False)
    breakpoint = 1


if __name__ =="__main__":
    test()


