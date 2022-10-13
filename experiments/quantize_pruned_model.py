import logging
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import entropy as compute_entropy
from torch.utils.data import DataLoader
from pactl.logging import set_logging
from pactl.random import random_seed_all
from pactl.data import get_dataset
from pactl.nn import create_model
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt
from pactl.prune_fns import get_pruned_vec
from pactl.prune_fns import recover_prune_mask
from pactl.bounds.quantize_fns import finetune_prune_quantization
from pactl.nn.projectors import _getchainattr
from pactl.bounds.compute_kl_mixture import get_gains


def compute_message_len(quantized_vec):
    uq, counts = np.unique(quantized_vec, return_counts=True)
    cluster_n = len(uq)
    nonzero_n = np.sum(np.where(uq == 0., 0., counts))
    message_len = nonzero_n * np.ceil(np.log(cluster_n))
    message_len += cluster_n * 16
    zero_loc_idx = np.argmax(uq == 0.0)
    symbols = assign_to_label(quantized_vec, uq)
    symbols = symbols[symbols != zero_loc_idx]
    probs = np.array([np.mean(symbols == i) for i in range(cluster_n)])
    probs = np.delete(probs, zero_loc_idx)
    entropy = compute_entropy(probs, base=2)
    message_len += np.ceil(len(symbols) * entropy) + 2
    message_len = message_len * np.log(2) + np.log(message_len)
    return message_len


def assign_to_label(vec, centroids):
    symbols = np.zeros(shape=vec.shape[0], dtype=np.int32)
    for i in range(centroids.shape[0]):
        mask = vec == centroids[i]
        symbols[mask] = i
    return symbols


@torch.no_grad()
def eval_acc(model, loader, scale=None, device=None, max_samples=5000):
    model.eval()
    if isinstance(scale, float):
        orig_weights = get_original_params(model)
        acc_samples = []
        std_err = np.inf
        while std_err > 3e-3:
            if len(acc_samples) >= max_samples:
                break

            for X, Y in tqdm(loader, desc='quantized acc eval'):
                X, Y = X.to(device), Y.to(device)
                update_params(model, orig_weights, scale)
                logits = model(X)
                acc = (logits.argmax(dim=-1) == Y).sum() / len(Y)
                acc_samples.append(acc.item())

            std_err = np.std(acc_samples) / np.sqrt(len(acc_samples))
        update_params(model, orig_weights, scale=0.)
        out = np.mean(acc_samples)
    else:
        correct, total = 0., 0.
        for X, Y in tqdm(loader, desc='evaluating model accuracy'):
            X, Y = X.to(device), Y.to(device)
            total += X.shape[0]
            labels = Y
            predictions = model(X).max(1)[1].type_as(labels)
            correct += predictions.eq(labels).cpu().data.numpy().sum()
        out = correct / total
    model.train()
    return out


def get_original_params(model):
    orig_weights = []
    for p in model.parameters():
        orig_weights.append(p.clone())
    return orig_weights


def update_params(model, orig_weights, scale):
    for idx, (name, param) in enumerate(model.named_parameters()):
        weight = orig_weights[idx].clone()
        if name.endswith("orig"):
            masked_name = name[:-4] + "mask"
            mask = _getchainattr(model, masked_name)
            param.data = weight + scale * torch.randn_like(weight)
            param.data *= mask
        else:
            param.data = weight + scale * torch.randn_like(weight)


def main(
    seed=137,
    device_id=0,
    data_dir=None,
    log_dir=None,
    dataset=None,
    prenet_cfg_path=None,
    batch_size=256,
    use_kmeans=False,
    levels=7,
    train_subset=1.,
    indices_path=None,
    quant_lr=1.e-4,
    quant_epochs=10,
    posterior_scale=0.1,
):

    random_seed_all(seed)
    device = torch.device(f"cuda:{device_id}")
    torch.backends.cudnn.benchmark = True
    log_dir = set_logging(log_dir=log_dir)
    train_data, test_data = get_dataset(
        dataset, root=data_dir,
        train_subset=train_subset,
        indices_path=indices_path)

    net = create_model(
        cfg_path=prenet_cfg_path, device_id=device_id, log_dir=log_dir
    )
    recover_prune_mask(net)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    initial_param_count = sum(p.numel() for p in net.parameters())
    print(f"Initial param count {initial_param_count:,d}")
    train_acc = eval_acc(net, train_loader, device=device)
    logging.info(f"Initial accuracy {train_acc:1.4e}")
    qw = finetune_prune_quantization(
        net, levels=levels - 1, device=device,
        train_loader=train_loader,
        epochs=quant_epochs,
        criterion=criterion,
        optimizer="sgd",
        lr=quant_lr,
        use_kmeans=False,
    )
    centroids = qw.centroids
    logging.info(centroids)
    vec = get_pruned_vec(net)
    vec = vec.detach().cpu().numpy()
    # posterior_scale = posterior_scale * np.std(vec)
    nonzero_param_count = initial_param_count - np.sum(vec == 0.)
    print(f"Nonzero param count {nonzero_param_count:,d}")
    print(f"Compression ratio {1 - nonzero_param_count / initial_param_count:1.3e}")
    message_len = compute_message_len(vec)
    logging.info(f"Message len {message_len}")
    train_acc = eval_acc(qw, train_loader, scale=float(posterior_scale), device=device)
    logging.info(f"Quantized accuracy {train_acc:1.4e}")

    print('*' * 50 + '\nEvaluating quantized accuracy and getting the bound')
    # TODO: is there a way to make != 0. safer?
    divergence_gains = get_gains(vec[vec != 0.], float(posterior_scale))
    message_len += divergence_gains

    err_bound = pac_bayes_bound_opt(
        divergence=message_len,
        train_error=1 - train_acc,
        n=len(train_loader.dataset)
    )
    logging.info(f"Error Bound {err_bound:1.7e}")

    bound_metrics = {
        'train_acc': train_acc,
        'message_len': message_len,
        'quant_train_err_100': (1 - train_acc) * 100,
        'err_bound_100': err_bound * 100,
    }
    logging.info(bound_metrics, extra=dict(wandb=True))


def entrypoint(**kwargs):
    main(**kwargs)


if __name__ == "__main__":
    import fire
    fire.Fire(entrypoint)
