import argparse
import time
import os

def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    # according to Yet-another-efficientdet-pytorch, the base lr 1e-4 if for total batchsize 12
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=48)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_save_sample', type=int, default=5)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--working_dir', type=str, default="./training_dir/")
    parser.add_argument('--path', type=str, default="data/")
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--result_file', type=str, default='./training_dir/')


    return parser

def get_args():
    parser = get_argparser()
    args = parser.parse_args()
    args.lr_steps = [8, 11]
    # args.lr_steps = [32, 44]
    args.lr_gamma = 0.1

    # backbone name type: 'model_type - coef' ,for examplt:'ResNet-101','Efficientdet-0'
    args.backbone_name = 'ResNet-50'
    get_mdoel_type(args)
    args.load_pretrained_weight = True
    args.load_checkpoint = False
    args.head_only = False
    args.weight_path = 'training_dir/2009171912/epoch-45.pt'
    #args.weight_path = 'weights/efficientdet-d0.pth'
    args.finetune = True
    args.early_stopping = False
    args.es_patience = 3
    args.val_with_loss = False
    args.lr_gamma_fpn = 1
    args.lr_gamma_backbone = 1

    args.lrschduler_type = 'cosine'
    args.warmup_epoch = 0.5

    args.EMA = False



    # args.feat_channels = [0, 0, 512, 768, 1024] # for vovnet
    # args.feat_channels = [0, 0, 128, 256, 512] # for resnet18
    args.feat_channels = [0, 0, 512, 1024, 2048] # for resnet50, resnet101
    args.out_channel = 256
    args.use_p5 = True
    #
    args.n_class = 16
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
    args.nms_threshold_mulit_scale = 0.8
    args.multi_scale_for_test = [(1024, 1024)]
    args.scale_weight = [[1, 1, 1]]
    args.object_size_threshold = [32*32, 96*96]
    args.flip_test_augmenation = True
    args.score_threshold_for_f1 = [0.4761747419834137, 0.5721955299377441, 0.0, 0.6717578172683716,
                            0.32183462381362915, 0.3168197274208069, 0.3369041979312897,
                            0.582460343837738, 0.5142934322357178, 0.0, 0.716709554195404,
                            0.6091494560241699, 0.32008489966392517, 0.38753411173820496, 0.4144534468650818]



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
    args.anchor_sizes = [32, 64, 128, 256, 512]
    args.anchor_strides = [8, 16, 32, 64, 128]
    args.anchor_ratios = [1.0]
    # topk for selecting candidate positive samples from each level

    # for loss
    args.reg_loss_weight = 2.0
    args.angle_loss_weight = 0.25
    args.cls_loss_weight = 2
    args.gamma = 2.0
    args.alpha = 0.25

    args.train_min_size_range = (640, 800)
    args.train_max_size = 1333
    args.test_min_size = 800
    args.test_max_size = 1333

    # for normalize
    args.pixel_mean = [0.319, 0.3258, 0.2951]
    args.pixel_std = [0.1245, 0.1224, 0.1173]

    # if you use the efficientdet, you don't have to modify this parameter
    args.size_divisible = 32

    adjust_working_dir(args)

    return args


def adjust_working_dir(args):
    base_working_dir = args.working_dir
    localtime = time.localtime()
    timestr = time.strftime('%y%m%d%H%M', localtime)
    args.working_dir = os.path.join(base_working_dir, timestr)
    args.result_file = os.path.join(base_working_dir, timestr)

def get_mdoel_type(args):
    model_name = args.backbone_name
    args.backbone_type = model_name.split('-')[0]
    args.backbone_coef = int(model_name.split('-')[-1])











