import logging
from pathlib import Path
import torch
from torch.optim import SGD
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from pactl.logging import set_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.nn import create_model
from pactl.prune_fns import prune_params
from pactl.prune_fns import make_prunning_permanent


def run_sgd(
    train_loader, net, criterion, optim, device, scheduler=None, epochs=0, log_dir=None
):

    for e in tqdm(range(epochs)):
        try:
            train_loader.sampler.set_epoch(e)
        except AttributeError:
            pass

        net.train()
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            optim.zero_grad()
            f_hat = net(X)
            loss = criterion(f_hat, Y)
            loss.backward()
            optim.step()

            if log_dir is not None and i % 100 == 0:
                metrics = {"epoch": e, "mini_loss": loss.detach().item()}
                logging.info(metrics, extra=dict(wandb=True, prefix="sgd/train"))

        if scheduler is not None:
            scheduler.step()


def eval_acc(model, loader, device):
    model.eval()
    with torch.no_grad():
        correct, total = 0.0, 0.0
        for X, Y in tqdm(loader, desc="evaluating model accuracy"):
            X, Y = X.to(device), Y.to(device)
            total += X.shape[0]
            labels = Y
            predictions = model(X).max(1)[1].type_as(labels)
            correct += predictions.eq(labels).cpu().data.numpy().sum()
        return correct / total


def main(
    seed=137,
    device_id=0,
    data_dir=None,
    log_dir=None,
    dataset=None,
    prenet_cfg_path=None,
    batch_size=256,
    lr=3e-3,
    train_subset=1.0,
    indices_path=None,
    amount=0.3,
    epochs=10,
    prune_iter=1,
):

    random_seed_all(seed)
    device = torch.device(f"cuda:{device_id}")
    torch.backends.cudnn.benchmark = True
    log_dir = set_logging(log_dir=log_dir)
    train_data, test_data = get_dataset(
        dataset, root=data_dir, train_subset=train_subset, indices_path=indices_path
    )

    net = create_model(cfg_path=prenet_cfg_path, device_id=device_id, log_dir=log_dir)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2)
    initial_param_count = sum(p.numel() for p in net.parameters())
    print(f"Initial param count {initial_param_count:,d}")
    train_acc = eval_acc(net, train_loader, device)
    print(f"Train acc: {train_acc:1.5e}")

    criterion = torch.nn.CrossEntropyLoss()
    for idx in range(prune_iter):
        optim = SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5.0e-4)
        sparsity = 1 - (1 - amount) ** (idx + 1)
        prune_params(net, amount=sparsity)
        run_sgd(train_loader, net, criterion, optim, device, epochs=epochs)
        train_acc = eval_acc(net, train_loader, device)
        logging.info(
            f"Prune Iter: {idx} | Prune Acc: {train_acc: 1.5e} "
            f"| Spars: {sparsity: 2.2f}"
        )
    make_prunning_permanent(net)
    torch.save(
        net.state_dict(), Path(log_dir) / "best_sgd_model.pt"
    )


def entrypoint(**kwargs):
    main(**kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(entrypoint)
