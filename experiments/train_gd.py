import logging
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from pactl.logging import set_logging, wandb, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.train_utils import eval_model
from pactl.nn import create_model

from pactl.optim.third_party.sgd_linesearch import RestartingLineSearch, NonMonotoneLinesearch, WolfeGradientDescent
from pactl.optim.third_party.optim_utils import get_scheduler


def run_gd(train_loader, test_loader, net, criterion, optim, scheduler=None, device_id=None,
            epochs=0, log_dir=None, max_norm=0.25):
  device = torch.device(f'cuda:{device_id}') if isinstance(device_id, int) else None
  
  best_acc_so_far = 0.

  for e in tqdm(range(epochs)):
    try:
      train_loader.sampler.set_epoch(e)
    except AttributeError:
      pass

    net.train()
    optim.zero_grad()
    for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
      X, Y = X.to(device), Y.to(device)


      f_hat = net(X)
      loss = criterion(f_hat, Y)

      loss.backward()

    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
    optim.step()

    if log_dir is not None:
      metrics = { 'epoch': e, 'loss': loss.detach().item() }
      logging.info(metrics, extra=dict(wandb=True, prefix='sgd/train'))

    if scheduler is not None:
      scheduler.step()

    if log_dir is not None:
      test_metrics = eval_model(net, test_loader, criterion, device_id=device)
      logging.info(test_metrics, extra=dict(wandb=True, prefix='sgd/test'))
      logging.info(test_metrics)

      if test_metrics['acc'] > best_acc_so_far:
        best_acc_so_far = test_metrics['acc']

        wandb.run.summary['sgd/test/best_epoch'] = e
        wandb.run.summary['sgd/test/best_acc'] = best_acc_so_far

        torch.save(net.state_dict(), Path(log_dir) / 'best_sgd_model.pt')

      torch.save(net.state_dict(), Path(log_dir) / 'sgd_model.pt')
      wandb.save('*.pt') ## NOTE: to upload immediately.


def main(seed=137, device_id=0, data_dir=None, log_dir=None,
         dataset=None, train_subset=1, label_noise=0, num_workers=2,
         cfg_path=None, transfer=False, model_name='resnet18k', base_width=None,
         batch_size=1000, optimizer='sgd', indices_path=None, scheduler_name="cosine-decay", lr=.4,
         momentum=.9, weight_decay=5e-4, epochs=0, nesterov=False, max_norm=0.25,
         intrinsic_dim=0, intrinsic_mode='sparse', warmup=0):

  random_seed_all(seed)
  torch.backends.cudnn.benchmark = True

  train_data, test_data = get_dataset(dataset, root=data_dir,
                                      train_subset=train_subset,
                                      indices_path=indices_path,
                                      label_noise=label_noise)

  net = create_model(model_name=model_name, num_classes=train_data.num_classes, in_chans=train_data[0][0].size(0), base_width=base_width,
                     seed=seed, intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode,
                     cfg_path=cfg_path, transfer=transfer, device_id=device_id, log_dir=log_dir)

  train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

  criterion = nn.CrossEntropyLoss()
  if optimizer == 'sgd':
    logging.info("------------ Using gradient descent ---------- ")
    optimizer = SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # optim_scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 
  elif optimizer == "sgd_wolfe": 
      optimizer = WolfeGradientDescent(net.parameters(), lr=lr, momentum=momentum, 
        weight_decay=weight_decay, nesterov=nesterov, dampening=0, c1=1e-4, c2=0.9,
        alpha_max=10.0, max_iter=10)
  elif optimizer == 'non-monotone':
      optimizer = NonMonotoneLinesearch(net.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=nesterov, dampening=0, interval=10,
        factor=0.25, max_iter=10)
  elif optimizer == 'restarting':
      optimizer = RestartingLineSearch(net.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=nesterov, dampening=0, interval=10,
        factor=0.25, max_iter=10)
  else:
    raise NotImplementedError
  optim_scheduler = get_scheduler(scheduler_name=scheduler_name,
      optimizer_to_schedule=optimizer, lr=lr, steps=epochs, warmup=warmup)
 
  run_gd(train_loader, test_loader, net, criterion, optimizer, scheduler=optim_scheduler, device_id=device_id,
          epochs=epochs, log_dir=log_dir, max_norm=max_norm)


def entrypoint(**kwargs):
  kwargs['log_dir'] = set_logging(log_dir=kwargs.get('log_dir'))
  main(**kwargs)
  finish_logging()


if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)
