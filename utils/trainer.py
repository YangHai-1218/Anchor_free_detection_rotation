from distributed import (
    get_rank,
    reduce_loss_dict,
    all_gather,
)
from tqdm import tqdm
import numpy as np
from torch import nn
import torch
from .test_augmenation import Multi_Scale_Test
from evaluate import evaluate


def accumulate_predictions(predictions):
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions





class Trainer:
    def __init__(self,args,loader,device):
        self.loader = loader
        self.device = device
        self.warmup_epoch = args.warmup_epoch
        self.n_gpu = args.n_gpu
        self.batch = args.batch

    def __call__(self, model, epoch,optimizer,scheduler=None,logger=None,ema=None):
        '''
        scheduler:[ scheduler, warmup scheduler]
        '''
        epoch_loss = []
        model.train()

        scheduler,warmup_scheduler = scheduler[0],scheduler[1]

        if get_rank() == 0:
            pbar = tqdm(enumerate(self.loader), total=len(self.loader), dynamic_ncols=True)
        else:
            pbar = enumerate(self.loader)

        for idx, (images, targets, _) in pbar:

            model.zero_grad()

            gt_exist = True
            for target in targets:
                gt_exist = gt_exist and len(target)
            if not gt_exist:
                continue
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]

            _, loss_dict = model(images, targets=targets)
            loss_cls = loss_dict['loss_cls'].mean()
            loss_box = loss_dict['loss_reg'].mean()
            loss_center = loss_dict['loss_centerness'].mean()

            loss = loss_cls + loss_box + loss_center

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            # ema update
            ema.update()

            # for iter scheduler
            if idx < warmup_scheduler.niters and epoch < self.warmup_epoch:
                warmup_scheduler.step()
            else:
                scheduler.step()

            loss_reduced = reduce_loss_dict(loss_dict)
            loss_cls = loss_reduced['loss_cls'].mean().item()
            loss_box = loss_reduced['loss_reg'].mean().item()
            loss_center = loss_reduced['loss_centerness'].mean().item()

            if get_rank() == 0:
                pbar.set_description(
                    (
                        f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                        f'box: {loss_box:.4f}; center: {loss_center:.4f}'
                    )
                )

                # writing log to tensorboard
                if logger and idx % 50 == 0:
                    lr_rate = optimizer.param_groups[0]['lr']
                    totalStep = (epoch * len(self.loader) + idx) * self.batch * self.n_gpu
                    logger.add_scalar('training/loss_cls', loss_cls, totalStep)
                    logger.add_scalar('training/loss_box', loss_box, totalStep)
                    logger.add_scalar('training/loss_center', loss_center, totalStep)
                    logger.add_scalar('training/loss_all', (loss_cls + loss_box + loss_center), totalStep)
                    logger.add_scalar('learning_rate', lr_rate, totalStep)

            epoch_loss.append(float(loss_cls + loss_box + loss_center))
        if logger:
            logger.add_scalar('training/loss_epoch_all', np.mean(epoch_loss), epoch)
        return epoch_loss


class Tester:
    def __init__(self,args,loader,dataset,device):
        self.distributed = args.distributed
        self.loader = loader
        self.dataset = dataset
        self.device = device

        self.multi_scale_test = args.multi_scale_test
        if self.multi_scale_test:
            self.multi_scale_tester = Multi_Scale_Test(self.dataset,
                                                  scale=args.multi_scale_for_test,
                                                  weight=args.scale_weight,
                                                  object_size_threshold=args.object_size_threshold,
                                                  nms_threshold=args.nms_threshold,
                                                  fpn_post_nms_top_n=args.detections_per_img,
                                                  device = self.device,
                                                  flip_enable=args.flip_test_augmenation,
                                                  bbox_aug_vote=args.voting_enable,
                                                  bbox_voting_threshold=args.voting_threshold,)


    @torch.no_grad()
    def __call__(self,model,epoch,logger=None,ema=None):
        if self.distributed:
            model = model.module

        torch.cuda.empty_cache()

        model.eval()

        if get_rank() == 0:
            pbar = tqdm(enumerate(self.loader), total=len(self.loader), dynamic_ncols=True)
        else:
            pbar = enumerate(self.loader)

        preds = {}

        for idx, (images, targets, ids) in pbar:
            model.zero_grad()

            images = images.to(self.device)

            if ema: ema.apply_shadow()

            if self.multi_scale_test:
                pred = self.multi_scale_tester(model,images,ids)
            else:
                pred, _ = model(images)

            if ema: ema.restore()

            pred = [p.to('cpu') for p in pred]

            preds.update({id: p for id, p in zip(ids, pred)})

        preds = accumulate_predictions(preds)

        if get_rank() != 0:
            return

        evl_res = evaluate(self.dataset, preds)

        # writing log to tensorboard
        if logger:
            log_group_name = "validation"
            box_result = evl_res['bbox']
            logger.add_scalar(log_group_name + '/AP', box_result['AP'], epoch)
            logger.add_scalar(log_group_name + '/AP50', box_result['AP50'], epoch)
            logger.add_scalar(log_group_name + '/AP75', box_result['AP75'], epoch)
            logger.add_scalar(log_group_name + '/APl', box_result['APl'], epoch)
            logger.add_scalar(log_group_name + '/APm', box_result['APm'], epoch)
            logger.add_scalar(log_group_name + '/APs', box_result['APs'], epoch)

        return preds