import argparse
import time
import os

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    # according to Yet-another-efficientdet-pytorch, the base lr 1e-4 if for total batchsize 12
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_save_sample', type=int, default=5)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--working_dir', type=str, default="./training_dir/")
    parser.add_argument('--path', type=str, default="data/DIOR/")


    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()
    args.lr_steps = [8, 11]
    # args.lr_steps = [32, 44]
    args.lr_gamma = 0.1

    # backbone name type: 'model_type - coef' ,for examplt:'ResNet-101','Efficientdet-0'
    args.backbone_name = 'Efficientdet-1'
    get_mdoel_type(args)
    args.load_pretrained_weight = False
    args.load_checkpoint = True
    args.weight_path = 'training_dir/2009010908/epoch-1.pt'
    args.head_only = False
    args.finetune = True
    args.early_stopping = False
    args.es_patience = 3
    args.val_with_loss = False
    args.lr_gamma_BiFPN = 0.1
    args.lr_gamma_Efficientnet= 0.1

    args.lrschduler_type = 'cosine'
    args.warmup_epoch = 0.5

    args.EMA = False



    # args.feat_channels = [0, 0, 512, 768, 1024] # for vovnet
    # args.feat_channels = [0, 0, 128, 256, 512] # for resnet18
    args.feat_channels = [0, 0, 512, 1024, 2048] # for resnet50, resnet101
    args.out_channel = 256
    args.use_p5 = True
    #
    args.n_class = 21
    args.n_conv = 4
    args.prior = 0.01

    # for inference
    args.inference_th = 0.05
    args.pre_nms_top_n = 1000
    args.nms_threshold = 0.6
    args.min_size = 0
    args.detections_per_img = 100
    args.voting_threshold = 0.5
    args.voting_enable = True
    args.multi_scale_test = False
    args.multi_scale_for_test = [(640,640),(896,896)]
    args.scale_weight = [[0.6,1,1],[1,1,0.6]]
    args.object_size_threshold = [32*32,96*96]
    args.flip_test_augmenation = True


    # for training
    # for assigner
    # how to select positves: ATSS , SSC (FCOS), IoU (RetinaNet), TOPK
    args.positive_type = "ATSS"
    args.top_k = 9
    args.fg_iou_threshold = 0.5
    args.bg_iou_threshold = 0.4
    # args.positive_type = "SSC"


    # regressing from a box ('BOX') or a point ('POINT')
    args.regression_type = "BOX"
    args.anchor_sizes = [64, 128, 256, 512, 1024]
    args.anchor_strides = [8, 16, 32, 64, 128]
    # topk for selecting candidate positive samples from each level

    # for loss
    args.reg_loss_weight = 2.0
    args.gamma = 2.0
    args.alpha = 0.25

    args.train_min_size_range = (640, 800)
    args.train_max_size = 1333
    args.test_min_size = 800
    args.test_max_size = 1333

    # for normalize
    args.pixel_mean = [0.485, 0.456, 0.406]
    args.pixel_std = [0.229, 0.224, 0.225]

    # if you use the efficientdet, you don't have to modify this parameter
    args.size_divisible = 32

    adjust_working_dir(args)

    return args


def adjust_working_dir(args):
    base_working_dir = args.working_dir
    localtime = time.localtime()
    timestr = time.strftime('%y%m%d%H%M',localtime)
    args.working_dir = os.path.join(base_working_dir,timestr)

def get_mdoel_type(args):
    model_name = args.backbone_name
    args.backbone_type = model_name.split('-')[0]
    args.backbone_coef = int(model_name.split('-')[-1])











