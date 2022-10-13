import os
import logging
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import csv


from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.train_utils import test_sample
from pactl.nn import create_model


def train(net, loader, criterion, optim, device=None, log_dir=None, epoch=None):
    net.train()

    for i, (X, Y) in tqdm(enumerate(loader), leave=False):
        X, Y = X.to(device), Y.to(device)

        optim.zero_grad()

        f_hat = net(X)
        loss = criterion(f_hat, Y)

        loss.backward()

        optim.step()

        if log_dir is not None and i % 100 == 0:
            metrics = { 'epoch': epoch, 'mini_loss': loss.detach().item() }
            logging.info(metrics, extra=dict(wandb=True, prefix='sgd/train'))


def main_train(seed=137, device_id=0, distributed=False, data_dir=None, log_dir=None,
         dataset=None, train_subset=1, indices_path=None, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='resnet18k', base_width=None,
         batch_size=128, optimizer_name='adam', lr=.1, momentum=.9, weight_decay=5e-4, epochs=0,
         intrinsic_dim=0, intrinsic_mode='filmrdkron'):

    random_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    train_data, test_data = get_dataset(
          dataset, root=data_dir,
          train_subset=train_subset,
          label_noise=label_noise,
          indices_path=indices_path)
    
    print("size of current subset training data: ", len(train_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                            shuffle=not distributed,
                            sampler=DistributedSampler(train_data) if distributed else None)
    ## FIXME: Fix reduce op for distributed eval before using distributed test loader.
        
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    net = create_model(model_name=model_name, num_classes=train_data.num_classes, in_chans=train_data[0][0].size(0), base_width=base_width,
                     seed=seed, intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode,
                     cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir)

    if torch.cuda.is_available():
        net.cuda()
    
    if distributed:
        # net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = nn.parallel.DistributedDataParallel(net, device_ids=[device_id], broadcast_buffers=True)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'sgd':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)    
    elif optimizer_name == 'adam':
        optimizer = Adam(net.parameters(), lr=lr)
        optim_scheduler = None
    else:
        raise NotImplementedError

    best_acc_so_far = 0.
    for e in tqdm(range(epochs)):
        if distributed:
            train_loader.sampler.set_epoch(e)

        train(net, train_loader, criterion, optimizer, device=device_id, log_dir=log_dir, epoch=e)

        if optim_scheduler is not None:
            optim_scheduler.step()

    test_metrics = test_sample(test_loader, net, criterion, device=device_id)
    subset_test_acc = test_metrics['acc']
    
    inverse_train_subset = -1 * train_subset
    
    train_data, test_data = get_dataset(
          dataset, root=data_dir,
          train_subset= inverse_train_subset,
          label_noise=label_noise,
          indices_path=indices_path)
    
    print("size of the rest of the training data: ", len(train_data))

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                            shuffle=not distributed,
                            sampler=DistributedSampler(train_data) if distributed else None)

    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == 'sgd':
        optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)    
    elif optimizer_name == 'adam':
        optimizer = Adam(net.parameters(), lr=lr)
        optim_scheduler = None
    else:
        raise NotImplementedError

    for e in tqdm(range(epochs)):
        if distributed:
            train_loader.sampler.set_epoch(e)

        train(net, train_loader, criterion, optimizer, device=device_id, log_dir=log_dir, epoch=e)

        if optim_scheduler is not None:
            optim_scheduler.step()

    test_metrics = test_sample(test_loader, net, criterion, device=device_id)
    test_acc = test_metrics['acc']
    
    return subset_test_acc, test_acc


def main(job_nb, device_id): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    indx_path = "./../cifar10_indx_train.npy"
    frac = 1/18
    eps = 1e-3
    train_subsets = np.linspace(frac * (job_nb-1) + eps, frac * job_nb - eps, 7)
    logname = "./results/cifar10_{}.csv".format(job_nb)
    print(logname)
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['train_subset', 'subset_test_acc', 'full_test_acc'])
    for train_subset in train_subsets: 
        subset_test_acc, full_test_acc = main_train(seed=137, device_id=device_id, distributed=False, data_dir=None, log_dir=None,
             dataset="cifar10", train_subset=train_subset, indices_path=indx_path, label_noise=0, num_workers=2,
             cfg_path=None, transfer=False, model_name='resnet18k', base_width=None,
             batch_size=128, optimizer_name='adam', lr=.1, momentum=.9, weight_decay=5e-4, epochs=100,
             intrinsic_dim=0, intrinsic_mode='filmrdkron')
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([train_subset, subset_test_acc, full_test_acc])
        
def entrypoint(**kwargs):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = 0
    device_id = 0
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        assert world_size == dist.get_world_size()

        rank = dist.get_rank()
        device_id = int(os.environ.get('LOCAL_RANK'))

#     kwargs['log_dir'] = set_logging(log_dir=kwargs.get('log_dir')) if rank == 0 else None
#     if rank == 0:
#         logging.info(f'Training with {world_size} process(es).')

    main(**kwargs, device_id=device_id)

#     if rank == 0:
#         finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)