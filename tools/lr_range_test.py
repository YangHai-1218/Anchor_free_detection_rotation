import torch
from torch import nn,optim
import cv2
from argument import get_args
import os
from tensorboardX import SummaryWriter
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    get_world_size,
)
from utils.dataset import DIORdataset,collate_fn
from utils import transform
from torch.utils.data import DataLoader, sampler
from train import data_sampler
from model import ATSS,Efficientnet_Bifpn_ATSS
from backbone import vovnet39, vovnet57, resnet50, resnet101
from utils.lrscheduler import GluonLRScheduler,iter_per_epoch_cal
from utils.trainer import Trainer,Tester


if __name__ == '__main__':
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    args = get_args()

    working_dir = args.working_dir + 'lr_range_test'
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    logger = SummaryWriter(working_dir)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        # torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = 'cuda'

    train_set = DIORdataset(args.path, 'train')
    train_trans = transform.Compose(
        [
            transform.RandomHorizontalFlip(0.5),
            # transform.Resize_For_Efficientnet(compund_coef=args.backbone_coef),
            transform.RandomMixUp(dataset=train_set),
            transform.Multi_Scale_with_Crop(scales=[640, 960, 1280], target_size=(640, 640)),
            transform.ToTensor(),
            transform.Normalize(args.pixel_mean, args.pixel_std),
        ]
    )
    train_set.set_transform(train_trans)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )

    valid_set = DIORdataset(args.path, 'val')
    valid_trans = transform.Compose(
        [
            transform.Resize_For_Efficientnet(compund_coef=args.backbone_coef),
            transform.ToTensor(),
            transform.Normalize(args.pixel_mean, args.pixel_std)
        ]
    )
    valid_set.set_transform(valid_trans)
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )

    if args.backbone_type == 'Efficientdet':
        if args.load_pretrained_weight:
            model = Efficientnet_Bifpn_ATSS(args,compound_coef=args.backbone_coef,load_backboe_weight=True,weight_path=args.weight_path)
        else:
            model = Efficientnet_Bifpn_ATSS(args,compound_coef=args.backbone_coef,load_backboe_weight=False)
    elif args.backbone_type == 'ResNet':
        if args.backbone_coef == 18:
            backbone = resnet50(pretrained=True)
        elif args.backbone_coef == 50:
            backbone = resnet50(pretrained=True)
        elif args.backbone_coef == 101:
            backbone = resnet101(pretrained=True)
        else:
            raise NotImplementedError(f'Not supported backbone name :{args.backbone_name}')
        model = ATSS(args, backbone)
    elif args.backbone_type == 'VovNet':
        if args.backbone_coef == 39:
            backbone = vovnet39(pretrained=True)
        elif args.backbone_coef == 57:
            backbone = vovnet57(pretrained=True)
        else:
            raise NotImplementedError(f'Not supported backbone name :{args.backbone_name}')
        model = ATSS(args, backbone)
    else:
        raise NotImplementedError(f'Not supported backbone name :{args.backbone_name}')

    model = model.to(device)

    if not args.head_only and args.finetune:
        # if not freeze the backbone, then finetune the backbone,
        optimizer = optim.SGD(
            model.backbone.backbone_net.parameters(),
            lr = 0,
            momentum = 0.9,
            weight_decay = 0.0001,
            nesterov = True,
        )
        optimizer.add_param_group({'params':list(model.backbone.bifpn.parameters()),'lr':0,
                                   'momentum': 0.9, 'weight_decay': 0.0001, 'nesterov': True})
        optimizer.add_param_group({'params':list(model.head.parameters()),'lr':0,'momentum':0.9,'weight_decay':0.0001,
                                   'nesterov':True})
        print(f'[INFO] efficientnet use the lr :{args.lr*args.lr_gamma_Efficientnet} to finetune,'
              f' bifpn use the lr:{args.lr*args.lr_gamma_BiFPN} to finetune')
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True,
        )

    iter_per_epoch = iter_per_epoch_cal(args, train_set)

    scheduler = GluonLRScheduler(optimizer, mode='linear', niters=int(iter_per_epoch*args.epoch),
                                        target_lr=[args.lr * args.lr_gamma_Efficientnet,
                                                   args.lr * args.lr_gamma_BiFPN,
                                                   args.lr])

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    trainer = Trainer(args,train_loader,device)
    valider = Tester(args,valid_loader,valid_set,device)

    print(f'[INFO] Start training: learning rate:{args.lr}, total batchsize:{args.batch * get_world_size()}, '
          f'working dir:{args.working_dir}')

    logger.add_text('exp_info', f'learning_rate:{args.lr},total_batchsize:{args.batch * get_world_size()},'
                                f'backbone_name:{args.backbone_name},freeze_backbone:{args.head_only},'
                                f'finetune_backbone:{args.finetune},lr_range_test:0 to {args.lr}')

    for epoch in range(args.epoch):
        epoch += 1

        epoch_loss = trainer(model,epoch,optimizer,scheduler=scheduler,logger=logger)
        valider(model,epoch,logger=logger)