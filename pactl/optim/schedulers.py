import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def construct_warm_stable_cosine(optimizer, lrs, epochs):
    lr_warm, lr_max, lr_min = lrs
    epochs1, epochs2, epochs3 = epochs

    def lr_lambda(epoch):
        if epoch <= epochs1:
            c1 = (lr_warm / lr_max)
            mult = c1 * ((epochs1 - epoch) / epochs1) + (epoch / epochs1)
        elif epoch <= epochs2 and epoch > epochs1:
            mult = 1.
        else:
            cons1 = lr_min / lr_max
            cons2 = 0.5 * ((lr_max - lr_min) / lr_max)
            cons3 = (1. + np.cos(((epoch - epochs2) / (epochs3 + epochs1)) * np.pi))
            mult = cons1 + cons2 * cons3
        return mult
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    return scheduler


def construct_stable_cosine(optimizer, lr_max, lr_min, epochs):
    epochs1, epochs2 = epochs

    def lr_lambda(epoch):
        if epoch <= epochs1:
            mult = 1.
        else:
            cons1 = lr_min / lr_max
            cons2 = 0.5 * ((lr_max - lr_min) / lr_max)
            cons3 = (1. + np.cos(((epoch - epochs1) / epochs2) * np.pi))
            mult = cons1 + cons2 * cons3
        return mult
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    return scheduler
