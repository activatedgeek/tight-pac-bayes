import unittest
import numpy as np
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR


class TestLRSchedulers(unittest.TestCase):

    def test_LambdaLR(self):
        lr = 4.
        epochs_n, decay_rate = 90, 0.01
        optimizer = get_setup(lr=lr)
        scheduler = construct_decay_every(optimizer, decay_rate, freq=30)

        lrs = np.zeros(epochs_n)
        for idx in range(epochs_n):
            lrs[idx] = scheduler.get_last_lr()[0]
            if idx % 30 == 0:
                print(f'lr: {scheduler.get_last_lr()} @ {idx}')
            optimizer.step()
            scheduler.step()
        print(f'lr: {scheduler.get_last_lr()} @ {idx}')

        print('TEST: LambdaLR: every 30 epochs decay')
        check = scheduler.get_last_lr()[0]
        aux = lr * (1. - decay_rate) ** (90 / 30)
        diff = np.abs(aux - check)
        print(f'Diff: {diff:1.3e}')
        self.assertTrue(expr=diff < 1.e-5)


def reconstruct_lr_schedule(optimizer, scheduler, epochs_n):
    lrs = np.zeros(shape=(epochs_n,))
    for idx in range(epochs_n):
        lrs[idx] = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
    return lrs


def construct_decay_every(optimizer, decay_rate, freq):
    def lr_lambda(epoch):
        return (1. - decay_rate) ** (epoch // freq)

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    return scheduler


def get_setup(lr):
    net = MLP()
    optimizer = SGD(net.parameters(), lr=lr)
    return optimizer


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Linear(5 * 5, 3)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
    unittest.main()
