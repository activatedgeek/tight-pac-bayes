import logging
from random import randint
from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from pactl.logging import set_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.bounds.quantize_fns import finetune_quantization
from pactl.nn import create_model


def eval_acc(model, loader, subsample=False):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for mb in tqdm(loader, desc="evaluating model accuracy"):
            total += mb[0].shape[0]
            labels = mb[1]
            predictions = model(mb[0].cuda()).max(1)[1].type_as(labels)
            correct += predictions.eq(labels).cpu().data.numpy().sum()
            if subsample and total > 5000:
                break
        return correct / total


def main(
    seed=137,
    device_id=0,
    data_dir=None,
    log_dir=None,
    dataset="fmnist",
    train_subset=1,
    label_noise=0,
    batch_size=128,
    optimizer="adam",
    lr=1.0e-2,
    momentum=0.9,
    weight_decay=5e-4,
    epochs=1,
    levels=3,
):

    random_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    log_dir = set_logging(log_dir=log_dir)

    train_data, test_data = get_dataset(
        dataset, root=data_dir, train_subset=train_subset, label_noise=label_noise
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2,
                              shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    model = create_model(
        cfg_path="./runs/net.cfg.yml", device_id=device_id, log_dir=log_dir
    )
    qw = finetune_quantization(
        model=model,
        levels=levels,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        epochs=epochs,
        device=torch.device(f"cuda:{device_id}"),
    )
    centroids = qw.centroids
    logging.info(centroids)
    acc = eval_acc(qw, test_loader)
    acc_train = eval_acc(qw, train_loader)
    logging.info(f"Test quantized accuracy {acc:1.4e}")
    logging.info(f"Train quantized accuracy {acc_train:1.4e}")
    results = {
        "epochs": epochs,
        "levels": levels,
        "test_acc": acc,
        "train_acc": acc_train,
    }
    df = pd.DataFrame([results])
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M_%S")
    file_path = "/res_" + time_stamp + "_" + str(randint(1, int(1.0e5))) + ".csv"
    df.to_csv(str(log_dir) + file_path, index=False)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
