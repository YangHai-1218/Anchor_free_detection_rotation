import torch
from torch.optim.lr_scheduler import _LRScheduler
import warnings
from math import cos,pi


class GluonLRScheduler(_LRScheduler):
    ''''
    gluon-cv:https://github.com/dmlc/gluon-cv/blob/3410d585a7c8b8e481911246efadfe5203853eb3/gluoncv/utils/lr_scheduler.py
    this lr sceduler adjust the learning rate according to iter, but not epoch,
    so if you want to use this, maske sure you are using it after every iter

    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    target_lr : float or list or tuple
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
        if you want to finetune the model, len(target_lr) should equal to the number of parameter groups
        and the order should equal to the order of using "optimizer.add_paramgroups"
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """
    '''
    def __init__(self,optimizer,mode,target_lr=0,
                 niters=0, nepochs=0, iters_per_epoch=0, offset=0,
                 power=2, step_iter=None, step_epoch=None, step_factor=0.1,
                 last_epoch=-1):

        if mode not in ['step','connstant','linear','poly','cosine','linear']:
            raise NotImplementedError
        self.mode = mode
        if mode == 'step':
            assert (step_iter is not None or step_epoch is not None)

        self.target_lr = target_lr
        self.niters = niters
        self.milestone = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.milestone = [s * iters_per_epoch for s in step_epoch]
        if offset == 0 and last_epoch !=-1:
            self.offset = (last_epoch+1)*epoch_iters
        else:
            self.offset = offset
        self.power = power
        self.step_factor = step_factor
        self.initial_flag = False
        super(GluonLRScheduler,self).__init__(optimizer,last_epoch)
        if self.mode == 'constant':
            self.target_lr = self.base_lrs



    def get_lr(self):
        N = self.niters - 1
        T = self._step_count - self.offset - 1
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + cos(pi * T / N)) / 2
        elif self.mode == 'step':
            if self.milestone is not None:
                count = sum([1 for s in self.milestone if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        if self.mode == 'step':
            learning_rate = [base_lr * factor for base_lr in self.base_lrs]
        else:
            if isinstance(self.target_lr, list) or isinstance(self.target_lr, tuple):
                # this is for finetune
                if len(self.target_lr) != len(self.optimizer.param_groups):
                    raise Exception("the number of target_lr should equal to the number of paramgroups")
                learning_rate = [(base_lr - target_lr) * factor + target_lr for base_lr, target_lr
                                 in zip(self.base_lrs, self.target_lr)]
            else:
                learning_rate = [(base_lr - self.target_lr) * factor + self.target_lr for base_lr in self.base_lrs]

        return learning_rate



def test():

    from torch.optim import SGD
    from argument import get_args
    from model import Efficientnet_Bifpn_ATSS
    args = get_args()
    model = Efficientnet_Bifpn_ATSS(args,load_backboe_weight=False)
    optimizer = SGD(model.parameters(),lr=0.1)
    optimizer.step()
    niters = int(1000)
    scheduler = GluonLRScheduler(optimizer,mode='linear',niters=niters,target_lr=0.2)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10)
    lrs = []
    lrs.append(optimizer.param_groups[0]['lr'])
    for i in range(niters):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])

    from matplotlib import pyplot as plt
    steps = [i for i in range(niters+1)]
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_axes([0.2,0.2,0.5,0.5])
    line, = ax.plot(steps,lrs)
    line.set_label('learning rate')
    plt.show()


if __name__ =='__main__':
    test()
