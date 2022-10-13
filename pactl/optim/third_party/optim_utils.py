"""This file is part of https://github.com/ildoonet/pytorch-gradual-warmup-lr.
MIT License
Copyright (c) 2019 Ildoo Kim
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

# @torch.no_grad()
# def _modify_gradient_params(norm_bias_strength):
#     if norm_bias_strength > 0.0:
#         param_norm_l2 = sum([p.pow(2).sum() for p in model.parameters()])
#         if cfg.hyp.norm_bias.norm_type == 1:
#             diff_value_sign = (param_norm_l2 - cfg.hyp.norm_bias.bias ** 2).sign()
#             [p.grad.add_(norm_bias_strength * diff_value_sign) for p in model.parameters()]
#         else:
#             factor = 2 * (param_norm_l2 - cfg.hyp.norm_bias.bias ** 2)
#             [p.grad.add_(norm_bias_strength * factor * p) for p in model.parameters()]

#     if cfg.hyp.grad_clip is not None:  # this is full clipping, we could also have block-level clipping
#         if cfg.hyp.grad_clip_norm == float('inf'):
#             grad_norm = max(p.grad.abs().max() for p in model.parameters())
#         else:
#             grad_norm = torch.norm(torch.stack([torch.norm(p.grad, cfg.hyp.grad_clip_norm) for p in model.parameters()]),
#                                    cfg.hyp.grad_clip_norm)
#         stats['preclip_gradnorm'] += [grad_norm.item()]
#         if grad_norm > cfg.hyp.grad_clip:
#             [p.grad.mul_(cfg.hyp.grad_clip / (grad_norm + 1e-6)) for p in model.parameters()]
#             log.info(f'Gradient total norm was {grad_norm}. Clipping to {cfg.hyp.grad_clip}.')
#             stats['clipped_step'] += [1]
#         else:
#             stats['clipped_step'] += [0]
#     if cfg.hyp.grad_noise['additive'] is not None:  # additive noise as in Langevin dynamics or diff. privacy
#         [p.grad.add_(cfg.hyp.grad_noise['additive'] * torch.randn_like(p)) for p in model.parameters()]
#     if cfg.hyp.grad_noise['multiplicative'] is not None:  # multiplicative noise as in Hoffer et al.
#         [p.grad.mul_(1 + cfg.hyp.grad_noise['multiplicative'] * torch.randn_like(p))
#          for p in model.parameters()]



def get_scheduler(scheduler_name, optimizer_to_schedule,
    lr, steps, warmup, multiplier=1.0):

    logging.info("---------- entering the get scheduler function -----------")

    if scheduler_name == 'linear':
        # Drop at 5/8, 6/8, 7/8:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_to_schedule,
                                                         milestones=[steps // 2.667, steps // 1.6,
                                                                     steps // 1.142], gamma=0.1)
    elif scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_to_schedule,
            gamma=0.99)
    elif scheduler_name == 'cosine-decay-floored':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_to_schedule, steps, eta_min=lr / 25)
    elif scheduler_name == 'cosine-decay':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule,
            steps, eta_min=0.0)
    elif scheduler_name == 'cosine-4000':
        # Cosine decay, hardcoded to 4000 steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_to_schedule,
            4000, eta_min=0.0)
    elif scheduler_name in ['', ' ', None]:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_to_schedule,
            milestones=[], gamma=1)
    else:
        raise ValueError(f'Invalid scheduler {scheduler_name} provided.')

    if warmup > 0:
        logging.info("---------------- warming up ----------------")
        scheduler = GradualWarmupScheduler(optimizer_to_schedule,
            multiplier=multiplier, total_epoch=warmup, after_scheduler=scheduler)

    return scheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        after_scheduler_dict = {key: value for key, value in self.after_scheduler.__dict__.items() if key != 'optimizer'}
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        state_dict['after_scheduler'] = after_scheduler_dict
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        after_scheduler_dict = state_dict.pop('after_scheduler')
        self.after_scheduler.__dict__.update(after_scheduler_dict)
        self.__dict__.update(state_dict)

