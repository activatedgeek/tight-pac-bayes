import logging
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pactl.train_utils import eval_model
from pactl.distributed import maybe_launch_distributed
from pactl.logging import set_logging, finish_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.nn import create_model
from pactl.nn.projectors import _delchainattr, _setchainattr
from pactl.bounds.quantize_fns import \
    get_random_symbols_and_codebook, \
    get_kmeans_symbols_and_codebook, \
    Quantize, \
    get_message_len, \
    do_arithmetic_encoding


class QuantizingWrapper(nn.Module):
    def __init__(self, net, quantizer, centroids):
        super().__init__()

        self._forward_net = [net]
        self.subspace_params = nn.Parameter(net.subspace_params.detach().clone(), requires_grad=True)
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=True)

        _delchainattr(self._forward_net[0], "subspace_params")

    def to(self, *args, **kwargs):
        self._forward_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        _setchainattr(
            self._forward_net[0],
            "subspace_params",
            self.quantizer(self.subspace_params, self.centroids),
        )
        return self._forward_net[0](*args, **kwargs)


def train(
    train_loader,
    net,
    criterion,
    optim,
    device=None,
    log_dir=None,
    epoch=None,
):

    net.train()
    for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        
        optim.zero_grad()
        
        f_hat = net(X)
        loss = criterion(f_hat, Y)
        
        loss.backward()
        
        optim.step()

        if log_dir is not None and i % 100 == 0:
            metrics = {"epoch": epoch, "mini_loss": loss.detach().item()}
            logging.info(metrics, extra=dict(wandb=True, prefix="sgd/train"))


def main(
    seed=137,
    device_id=0,
    data_dir=None,
    log_dir=None,
    dataset=None,
    prenet_cfg_path=None,
    batch_size=256,
    optimizer='sgd',
    lr=3e-3,
    use_kmeans=False,
    levels=7,
    epochs=10,
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
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=not distributed,
        sampler=DistributedSampler(train_data) if distributed else None)
    # test_loader = DataLoader(
    #     test_data,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     sampler=DistributedSampler(test_data) if distributed else None)

    model = create_model(cfg_path=prenet_cfg_path,
                       device_id=device_id,
                       log_dir=log_dir)

    cluster_fn = get_kmeans_symbols_and_codebook if use_kmeans else get_random_symbols_and_codebook
    _, centroids = cluster_fn(model.subspace_params.cpu().data.numpy(), levels=levels, codebook_dtype=np.float16)
    centroids = torch.from_numpy(centroids).float()
    
    qw = QuantizingWrapper(model, quantizer=Quantize().apply, centroids=centroids).to(device_id)
    if distributed:
        qw = torch.nn.parallel.DistributedDataParallel(qw, device_ids=[device_id], broadcast_buffers=True)

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(qw.parameters(), lr=lr)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(qw.parameters(), lr=lr)
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss()
    for e in tqdm(range(epochs)):
        if distributed:
            train_loader.sampler.set_epoch(e)

        train(
            train_loader,
            qw,
            criterion,
            optimizer,
            device=device_id,
            log_dir=log_dir,
            epoch=e,
        )

        train_acc = eval_model(model, train_loader, device_id=device_id, distributed=distributed)['acc']
        # test_acc = eval_model(model, test_loader, device_id=device_id, distributed=distributed)['acc']

        logging.info(f'Epoch {e}: {train_acc:.4f} (Train)')
        # logging.info(f'Epoch {e}: {train_acc:.4f} (Train) / {test_acc:.4f} (Test)')

    quantized_vec = qw.quantizer(qw.subspace_params, qw.centroids)
    quantized_vec = quantized_vec.cpu().detach().numpy()
    vec = (qw.centroids.unsqueeze(-2) - qw.subspace_params.unsqueeze(-1))**2.0
    symbols = torch.min(vec, -1)[-1]
    symbols = symbols.cpu().detach().numpy()
    centroids = qw.centroids.cpu().detach().numpy()
    probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
    _, coded_symbols_size = do_arithmetic_encoding(symbols, probabilities,
                                                    qw.centroids.shape[0])
    message_len = get_message_len(
        coded_symbols_size=coded_symbols_size,
        codebook=centroids,
        max_count=len(symbols),
    )

    with open('ckpt.pt', 'wb') as f:
        torch.save({
            'qvec': quantized_vec,
            'message_len': message_len,
            'train_acc': train_acc,
        }, f)




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
