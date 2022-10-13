import logging
import yaml
from pathlib import Path
import timm
import torch

from ..logging import wandb
from .projectors import create_intrinsic_model


def create_model(model_name=None, num_classes=None, base_width=None, in_chans=None,
                 seed=None, intrinsic_dim=0, intrinsic_mode='sparse',
                 cfg_path=None, transfer=False, device_id=None, log_dir=None):

  device = torch.device(f'cuda:{device_id}') if isinstance(device_id, int) else None

  ## Prepare configurations.
  net_cfg, intrinsic_cfg = None, None
  base_ckpt_path, id_ckpt_path = None, None

  if cfg_path is not None:
    with open(cfg_path, 'r') as f:
      net_cfg = yaml.safe_load(f)
    
    intrinsic_cfg = net_cfg.get('intrinsic')
    if intrinsic_cfg is not None:
      net_cfg.pop('intrinsic')

    net_ckpt_file = 'init_model.pt' if intrinsic_cfg is not None else \
                    net_cfg.get('ckpt_file', 'best_sgd_model.pt')
    base_ckpt_path = Path(cfg_path).parent / net_ckpt_file

    id_ckpt_file = net_cfg.get('ckpt_file', 'best_sgd_model.pt') if intrinsic_cfg is not None else\
                   None
    id_ckpt_path = Path(cfg_path).parent / id_ckpt_file if id_ckpt_file is not None else None
  else:
    net_cfg = dict(model_name=model_name, num_classes=num_classes, in_chans=in_chans)
    if base_width is not None:
      net_cfg['base_width'] = base_width

  ## Always try setup, but avoid overriding existing intrinsic config.
  if intrinsic_cfg is None and intrinsic_dim > 0:
    intrinsic_cfg = dict(intrinsic_dim=intrinsic_dim, intrinsic_mode=intrinsic_mode, seed=seed)

  ## Load base model.
  base_net = timm.create_model(**net_cfg, checkpoint_path=base_ckpt_path)
  if base_ckpt_path is not None:
    logging.info(f'Loaded base model from "{base_ckpt_path}".')

  ## Replace classifier head for transfer learning.
  if transfer:
    base_net.reset_classifier(num_classes)
    net_cfg['num_classes'] = num_classes
    logging.info(f'Reset classifier for {num_classes} classes.')

  base_net = base_net.to(device)
  if log_dir is not None:
    ## Save initialization.
    torch.save(base_net.state_dict(), Path(log_dir) / 'init_model.pt')
    logging.info(f'Saved base model at "{Path(log_dir) / "init_model.pt"}".')

  ## Create intrinsic dimensionality model.
  final_net = base_net if intrinsic_cfg is None else \
              create_intrinsic_model(base_net, **intrinsic_cfg, ckpt_path=id_ckpt_path, device=device)
  final_net = final_net.to(device)
  if id_ckpt_path is not None:
    logging.info(f'Loaded ID model from "{id_ckpt_path}".')
  
  ## Bookkeeping.
  if log_dir is not None:
    dump_cfg = dict(**net_cfg)
    if intrinsic_cfg is not None:
      dump_cfg['intrinsic'] = intrinsic_cfg
    with open(Path(log_dir) / 'net.cfg.yml', 'w') as f:
      yaml.safe_dump(dump_cfg, f, indent=2)
    logging.info(f'Saved net configuration at "{Path(log_dir) / "net.cfg.yml"}".')

    wandb.config.update({**net_cfg, **(intrinsic_cfg or dict()) })
    wandb.save('*.yml')

  return final_net
