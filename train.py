import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm
from argument import get_args
from backbone import vovnet39, vovnet57, resnet50, resnet101
from utils.dataset import DOTADataset, collate_fn
from model import ATSS,Efficientnet_Bifpn_ATSS
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    get_world_size,
)
from utils import (
    EMA,
    Trainer,
    Tester,
    GluonLRScheduler,
    transform,
    iter_per_epoch_cal,
    set_schduler_with_wormup
)
import os, cv2
from tensorboardX import SummaryWriter
import numpy as np




@torch.no_grad()
def valid_loss(args, epoch, loader, dataset, model, device, logger=None):
    loss_regress_list = []
    loss_cls_list = []
    loss_centerness_list = []

    if args.distributed:
        model = model.module

    torch.cuda.empty_cache()

    model.eval()

    if get_rank() == 0:
        pbar = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=True)
    else:
        pbar = enumerate(loader)

    preds = {}

    for idx, (images, targets, ids) in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        pred,loss_dict = model(images,targets,args.val_with_loss)


        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_reg'].mean().item()
        loss_center = loss_reduced['loss_centerness'].mean().item()
        loss_regress_list.append(float(loss_box))
        loss_cls_list.append(float(loss_cls))
        loss_centerness_list.append(float(loss_center))


    if logger:
        log_group_name = 'validation'
        logger.add_scalar(log_group_name+'/class_loss',np.mean(loss_cls_list),epoch)
        logger.add_scalar(log_group_name+'/regression_loss',np.mean(loss_regress_list),epoch)
        logger.add_scalar(log_group_name+'/centerness_loss',np.mean(loss_centerness_list),epoch)
        loss_all = np.mean(loss_cls_list) + np.mean(loss_regress_list) + np.mean(loss_centerness_list)
        logger.add_scalar(log_group_name+'/loss_epoch_all',loss_all,epoch)
    return loss_all





def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)



def save_checkpoint(model,args,optimizer,epoch):
    if get_rank() == 0:
        if args.distributed:
            torch.save(
                {'model': model.module.state_dict(), 'optim': optimizer.state_dict()},
                args.working_dir + f'/epoch-{epoch + 1}.pt',
            )
        else:
            torch.save(
                {'model': model.state_dict(), 'optim': optimizer.state_dict()},
                args.working_dir + f'/epoch-{epoch + 1}.pt',
            )


if __name__ == '__main__':

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    args = get_args()

    # Create working directory for saving intermediate results
    working_dir = args.working_dir
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    logger = SummaryWriter(working_dir)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.n_gpu = n_gpu
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='gloo', init_method='env://')
        #torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = 'cuda'

    train_set = DOTADataset(path=args.path, split='train', image_folder_name='min_split_',
                            anno_folder_name='annotations_split_')
    train_trans = transform.Compose(
        [
            transform.RandomHorizontalFlip(0.5),
            #transform.RandomMixUp(dataset=train_set),
            transform.RandomVerticalFlip(0.5),
            transform.RandomRotate(0.5, rotate_time=4),
            transform.Cutout(0.5),
            transform.Multi_Scale_with_Crop(scales=[768, 896, 1024, 1152], target_size=(768, 768)),
            transform.ToTensor(),
            transform.Normalize(args.pixel_mean, args.pixel_std),
        ]
    )
    train_set.set_transform(train_trans)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        sampler=data_sampler(train_set, shuffle=False, distributed=args.distributed),
        num_workers=args.num_workers,
        collate_fn=collate_fn(args),
    )

    # valid_set = DIORdataset(args.path, 'val')
    # valid_trans = transform.Compose(
    #     [
    #         transform.Resize_For_Efficientnet(compund_coef=args.backbone_coef),
    #         transform.ToTensor(),
    #         transform.Normalize(args.pixel_mean, args.pixel_std)
    #     ]
    # )
    # valid_set.set_transform(valid_trans)
    # valid_loader = DataLoader(
    #     valid_set,
    #     batch_size=args.batch,
    #     sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
    #     num_workers=args.num_workers,
    #     collate_fn=collate_fn(args),
    # )


    # if args.val_with_loss:
    #     valid_loss_set = COCODataset(args.path, 'val_loss', valid_trans)
    #     val_loss_loader = DataLoader(
    #         valid_loss_set,
    #         batch_size=args.batch,
    #         sampler=data_sampler(valid_loss_set, shuffle=False, distributed=args.distributed),
    #         num_workers=args.num_workers,
    #         collate_fn=collate_fn(args),
    #     )




    # backbone = vovnet39(pretrained=True)
    # backbone = vovnet57(pretrained=True)
    # backbone = resnet18(pretrained=True)
    # backbone = resnet50(pretrained=True)
    # backbone = resnet101(pretrained=True)
    # model = ATSS(args, backbone)

    if args.backbone_type == 'Efficientdet':
        if args.load_pretrained_weight:
            model = Efficientnet_Bifpn_ATSS(args, compound_coef=args.backbone_coef,
                                            load_backboe_weight=True, weight_path=args.weight_path)
        else:
            model = Efficientnet_Bifpn_ATSS(args, compound_coef=args.backbone_coef,
                                            load_backboe_weight=False)
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

    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.weight_path,map_location='cpu')['model'])
        print(f'[INFO] load checkpoint weight successfully!')


    # freeze backbone and FPN if train head_only
    if args.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN','FPN','FPNTopP6P7','ResNet']:
                if ntl == classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')



    if not args.head_only and args.finetune:
        # if not freeze the backbone, then finetune the backbone,
        optimizer = optim.SGD(
            model.backbone.backbone_net.parameters(),
            lr=0,
            momentum=0.9,
            weight_decay=0.0001,
            nesterov=True,
        )
        optimizer.add_param_group({'params': list(model.backbone.bifpn.parameters()), 'lr': 0,
                                   'momentum': 0.9, 'weight_decay': 0.0001, 'nesterov': True})
        optimizer.add_param_group({'params': list(model.head.parameters()), 'lr': 0,
                                   'momentum': 0.9, 'weight_decay': 0.0001, 'nesterov': True})
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


    if args.load_checkpoint:
        optimizer.load_state_dict(torch.load(args.weight_path)['optim'])
        last_epoch = int(os.path.basename(args.weight_path).split('.')[0][6:])
        print(f'[INFO] load optimizer state:{last_epoch}')
        last_epoch = last_epoch - 1
    else:
        last_epoch = -1


    # scheduler = optim.lr_scheduler.MultiStepLR(
    #     optimizer, milestones=args.lr_steps, gamma=args.lr_gamma,last_epoch=last_epoch
    # )
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.lr_gamma, patience=3,verbose=True)
    iter_per_epoch = iter_per_epoch_cal(args, train_set)
    scheduler = GluonLRScheduler(optimizer, mode='cosine', nepochs=(args.epoch-args.warmup_epoch),
                                 iters_per_epoch=iter_per_epoch)
    warmup_scheduler, schdeduler = set_schduler_with_wormup(args, iter_per_epoch, optimizer, scheduler)


    ema = EMA(model,decay=0.999,enable=args.EMA)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    trainer = Trainer(args, train_loader, device)
    #valider = Tester(args,valid_loader,valid_set,device)

    print(f'[INFO] Start training: learning rate:{args.lr}, total batchsize:{args.batch*get_world_size()}, '
          f'working dir:{args.working_dir}')

    logger.add_text('exp_info',f'learning_rate:{args.lr},total_batchsize:{args.batch*get_world_size()},'
                               f'backbone_name:{args.backbone_name},freeze_backbone:{args.head_only},'
                               f'finetune_backbone:{args.finetune}')

    if args.finetune:
        logger.add_text('exp_info',f'efficientnet lr gamma:{args.lr_gamma_Efficientnet},'
                                   f'BiFPN lr gamma:{args.lr_gamma_BiFPN}')

    val_best_loss = 1e5
    val_best_epoch = 0

    for epoch in range(args.epoch-(last_epoch+1)):
        epoch += (last_epoch + 1)

        epoch_loss = trainer.train(model, epoch, optimizer, [scheduler, warmup_scheduler], logger, ema)

        save_checkpoint(model, args, optimizer, epoch)

        #valider(model,epoch,logger,ema)

        # if args.val_with_loss and epoch > 1 and epoch % 2 ==0:
        #     val_epoch_loss = valid_loss(args,epoch,val_loss_loader,valid_loss_set,model,device,logger=logger)
        #
        #     if args.early_stopping :
        #         if val_epoch_loss < val_best_loss:
        #             val_best_loss = val_epoch_loss
        #             val_best_epoch = epoch
        #
        #         if epoch - val_best_epoch > args.es_patience:
        #             print(f'[INFO]Stop training at epoch {epoch}. The lowest validation loss achieved is {val_best_loss}')
        #             save_checkpoint(model,args,optimizer,epoch)



        # scheduler.step(np.mean(epoch_loss))