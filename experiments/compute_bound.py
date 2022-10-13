import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.nn import create_model
from pactl.bounds.get_bound_from_chk_v2 import evaluate_idmodel


def main(
    seed=137,
    device_id=0,
    data_dir=None,
    log_dir=None,
    dataset=None,
    prenet_cfg_path=None,
    batch_size=256,
    lr=3e-3,
    use_kmeans=False,
    levels=7,
    posterior_scale=0.1,
    misc_extra_bits=0,
    quant_epochs=10,
    encoding_type='arithmetic',
    train_subset=1.,
    indices_path=None,
    num_workers=4,
    distributed=False,
):

    random_seed_all(seed)

    train_data, test_data = get_dataset(dataset,
                                        root=data_dir,
                                        train_subset=train_subset,
                                        indices_path=indices_path)

    net = create_model(cfg_path=prenet_cfg_path,
                       device_id=device_id,
                       log_dir=log_dir)
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net,
                                                        device_ids=[device_id],
                                                        broadcast_buffers=True)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not distributed,
        sampler=DistributedSampler(train_data) if distributed else None)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=DistributedSampler(test_data) if distributed else None)

    bound_metrics = evaluate_idmodel(
        net,
        train_loader,
        test_loader,
        use_kmeans=bool(use_kmeans),
        levels=levels,
        device=torch.device(f"cuda:{device_id}"),
        lr=lr,
        epochs=quant_epochs,
        posterior_scale=posterior_scale,
        misc_extra_bits=misc_extra_bits,
        distributed=distributed,
        log_dir=log_dir,
    )
    if log_dir is not None:
        logging.info(bound_metrics, extra=dict(wandb=True))


def entrypoint(log_dir=None, **kwargs):
    world_size, rank, device_id = maybe_launch_distributed()

    if 'device_id' in list(kwargs.keys()):
        device_id = kwargs['device_id']
        kwargs.pop('device_id')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(device_id)

    ## Only setup logging from one process (rank = 0).
    log_dir = set_logging(log_dir=log_dir) if rank == 0 else None
    if rank == 0:
        logging.info(f'Working with {world_size} process(es).')

    main(**kwargs,
         log_dir=log_dir,
         distributed=(world_size > 1),
         device_id=device_id)

    if rank == 0:
        finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)
