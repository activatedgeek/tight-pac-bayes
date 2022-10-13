from matplotlib import pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from tests.TestLRSchedulers import get_setup
from tests.TestLRSchedulers import construct_decay_every
from tests.TestLRSchedulers import reconstruct_lr_schedule
from pactl.optim.schedulers import construct_stable_cosine
from pactl.optim.schedulers import construct_warm_stable_cosine


epochs_n = int(4e2)
lr = 5e-0
optimizer = get_setup(lr=lr)
scheduler5 = construct_warm_stable_cosine(
    optimizer, lrs=(lr/100., lr, lr/100.), epochs=(20, 100, epochs_n - 120))
lr5 = reconstruct_lr_schedule(optimizer, scheduler5, epochs_n=epochs_n)

plt.style.use('ggplot')
plt.figure(dpi=150, figsize=(14, 8))
steps = np.arange(epochs_n) + 1.
plt.plot(steps, lr5)
plt.show()

epochs_n = int(4e2)
lr = 5e-0
optimizer = get_setup(lr=lr)
scheduler4 = construct_stable_cosine(optimizer, lr, lr/100., (100, epochs_n - 100))
lr4 = reconstruct_lr_schedule(optimizer, scheduler4, epochs_n=epochs_n)

plt.style.use('ggplot')
plt.figure(dpi=150, figsize=(14, 8))
steps = np.arange(epochs_n) + 1.
plt.plot(steps, lr4)
plt.show()

epochs_n = int(1e2)
lr = 1e-1
decay_rate = 0.001
optimizer = get_setup(lr=lr)
scheduler1 = ExponentialLR(optimizer, gamma=1. - decay_rate)
lr1 = reconstruct_lr_schedule(optimizer, scheduler1, epochs_n)
scheduler2 = construct_decay_every(optimizer, decay_rate, freq=1)
lr2 = reconstruct_lr_schedule(optimizer, scheduler2, epochs_n)

plt.style.use('ggplot')
plt.figure(dpi=150, figsize=(14, 8))
steps = np.arange(epochs_n) + 1.
plt.plot(steps, lr1, label='ExponentialLR')
plt.plot(steps, lr2, label='Mine')
plt.show()

plt.style.use('ggplot')
plt.figure(dpi=150, figsize=(14, 8))
steps = np.arange(epochs_n) + 1.
plt.plot(steps, lr1 - lr2, label='diff')
plt.show()

epochs_n = int(1e3)
lr = 1e-1
optimizer = get_setup(lr=lr)
scheduler3 = CosineAnnealingLR(optimizer, T_max=epochs_n)
lr3 = reconstruct_lr_schedule(optimizer, scheduler3, epochs_n=epochs_n)

plt.style.use('ggplot')
plt.figure(dpi=150, figsize=(14, 8))
steps = np.arange(epochs_n) + 1.
plt.plot(steps, lr3, label='diff')
plt.show()
