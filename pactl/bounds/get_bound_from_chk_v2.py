import os
import copy
import dill
import logging
from datetime import datetime
from random import randint
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm.auto import tqdm
from pactl.nn.projectors import FixedNumpySeed, FixedPytorchSeed
from pactl.bounds.quantize_fns import quantize_vector
from pactl.bounds.quantize_fns import finetune_quantization
from pactl.bounds.quantize_fns import get_message_len
from pactl.bounds.quantize_fns import do_arithmetic_encoding
from pactl.bounds.get_pac_bounds import pac_bayes_bound_opt
from pactl.bounds.get_pac_bounds import compute_catoni_bound
from pactl.train_utils import eval_model, DistributedValue
from pactl.bounds.compute_kl_mixture import get_gains
from pactl.data import get_dataset, get_data_dir
import torch.nn.functional as F

# def eval_acc(model, loader, subsample=False):
#     model.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for x, y in tqdm(loader, desc='evaluating model accuracy'):
#             total += x.shape[0]
#             labels = y
#             predictions = model(x.cuda()).max(1)[1].type_as(labels)
#             correct += predictions.eq(labels).cpu().data.numpy().sum()
#             if subsample and total > 5000:
#                 break
#         return correct/total


@torch.no_grad()
def eval_perturbed_model(
    model,
    loader,
    scale=None,
    device=None,
    max_samples=5000,
    distributed=False,
):
    assert scale is not None

    model.eval()

    module = model.module if distributed else model

    orig_weights = copy.deepcopy(module.subspace_params.data)

    acc_samples = []
    std_err = np.inf
    if distributed:
        acc_samples = DistributedValue(acc_samples)

    while std_err > 3e-3:
        for X, Y in tqdm(loader, desc='quantized acc eval'):
            module.subspace_params.data = orig_weights + \
                scale * torch.randn_like(orig_weights)

            X, Y = X.to(device), Y.to(device)

            logits = model(X)

            acc = (logits.argmax(dim=-1) == Y).sum() / len(Y)

            acc_samples += [acc.item()]

        if distributed:
            acc_samples = acc_samples.resolve()

        std_err = np.std(acc_samples) / np.sqrt(len(acc_samples))

        if len(acc_samples) >= max_samples:
            break

        if distributed:
            acc_samples = DistributedValue(acc_samples)

    if distributed:
        acc_samples = acc_samples.resolve()

    module.subspace_params.data = orig_weights
    out = np.mean(acc_samples)

    return out


def compute_quantization(
    model,
    levels,
    device,
    train_loader,
    epochs,
    lr,
    use_kmeans,
):
    if levels == 0:
        return None, 0

    use_finetuning = True if epochs > 0 else False
    if use_finetuning:
        ## FIXME: for distributed training.
        criterion = nn.CrossEntropyLoss()
        qw = finetune_quantization(
            model=model,
            levels=levels,
            device=device,
            train_loader=train_loader,
            epochs=epochs,
            criterion=criterion,
            # optimizer='adam',
            optimizer='sgd',
            lr=lr,
            use_kmeans=use_kmeans,
        )
        quantized_vec = qw.quantizer(qw.subspace_params, qw.centroids)
        quantized_vec = quantized_vec.cpu().detach().numpy()
        vec = (qw.centroids.unsqueeze(-2) - qw.subspace_params.unsqueeze(-1))**2.0
        symbols = torch.min(vec, -1)[-1]
        symbols = symbols.cpu().detach().numpy()
        centroids = qw.centroids.cpu().detach().numpy()
        # centroids = centroids.astype(np.float16)
        probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
        _, coded_symbols_size = do_arithmetic_encoding(symbols, probabilities,
                                                       qw.centroids.shape[0])
        message_len = get_message_len(
            coded_symbols_size=coded_symbols_size,
            codebook=centroids,
            max_count=len(symbols),
        )
    else:
        module = model.module if isinstance(model,
                                            torch.nn.parallel.DistributedDataParallel) else model
        vector = module.subspace_params.cpu().data.numpy()
        quantized_vec, message_len = quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)
    return quantized_vec, message_len


def evaluate_idmodel(
    model,
    trainloader,
    testloader,
    lr=1.0e-2,
    epochs=10,
    device=torch.device('cuda'),
    posterior_scale=0.01,
    misc_extra_bits=0,
    levels=7,
    use_kmeans=False,
    distributed=False,
    log_dir=None,
):

    train_acc = eval_model(model, trainloader, device_id=device, distributed=distributed)['acc']
    if log_dir is not None:
        logging.info(f'Train accuracy: {train_acc:.4f}')
    test_acc = eval_model(model, testloader, device_id=device, distributed=distributed)['acc']
    if log_dir is not None:
        logging.info(f'Test accuracy: {test_acc:.4f}')

    quantized_vec, message_len = compute_quantization(model, levels, device, trainloader, epochs,
                                                      lr, use_kmeans)

    try:
        module = model.module if distributed else model
        # TODO: Ideally use PyTorch parameters instead of private variable.
        if quantized_vec is not None:
            module.subspace_params.data = torch.tensor(quantized_vec).float().to(device)
        else:
            aux = torch.zeros_like(module.subspace_params.data).float().to(device)
            module.subspace_params.data = aux
    except AttributeError:
        logging.warning("Quantization vector was not updated.")

    # self delimiting message takes up additional 2 log(l) bits
    prefix_message_len = message_len + 2 * np.log2(message_len) if message_len > 0 else 0
    train_nll_bits = total_nll_bits(model, trainloader, device=device, distributed=distributed)
    # output = eval_acc_and_bound(
    #     model=model, trainloader=trainloader, testloader=testloader,
    #     prefix_message_len=prefix_message_len,
    #     device=device,
    #     quantized_vec=quantized_vec, posterior_scale=posterior_scale)

    raw_output = eval_acc_and_bound(model=model, trainloader=trainloader, testloader=testloader,
                                    prefix_message_len=prefix_message_len, device=device,
                                    misc_extra_bits=misc_extra_bits, quantized_vec=quantized_vec,
                                    posterior_scale=posterior_scale, use_robust_adj=False,
                                    log_dir=log_dir, distributed=distributed)
    raw_output = {f'raw_{name}': value for name, value in raw_output.items()}
    return {
        # **output,
        **raw_output,
        'train_nll_bits': train_nll_bits,
        'prefix_message_len': prefix_message_len,
        'train_err_100': (1. - train_acc) * 100,
        'test_err_100': (1 - test_acc) * 100
    }


@torch.no_grad()
def total_nll_bits(
    model,
    loader,
    device=None,
    distributed=False,
):
    model.eval()

    nll = torch.tensor(0.).to(device)
    if distributed:
        nll = DistributedValue(nll)

    for x, y in tqdm(loader, desc='evaluating nll', leave=False):
        # compute log probabilities and index them by the labels
        logprobs = model(x.to(device)).log_softmax(dim=1)[np.arange(y.shape[0]), y]
        nll += -logprobs.sum().cpu().data.item() / np.log(2)

    if distributed:
        nll = nll.resolve()

    return nll


def eval_acc_and_bound(
    model,
    trainloader,
    testloader,
    prefix_message_len,
    quantized_vec,
    posterior_scale=0.01,
    misc_extra_bits=0.,
    device=None,
    use_robust_adj=True,
    log_dir=None,
    distributed=False,
):
    if log_dir is not None:
        logging.debug('*' * 50 + '\nEvaluating quantized accuracy and getting the bound')

    posterior_scale = None if quantized_vec is None else posterior_scale * np.std(quantized_vec)

    # assert posterior_scale is not None

    if use_robust_adj:
        divergence_gains = get_gains(quantized_vec, posterior_scale)
    else:
        divergence_gains = 0
        # NOTE: Don't pay for posterior scales (we optimize over 4 scales).
        misc_extra_bits -= 2
        posterior_scale = None

    if posterior_scale is None:
        quant_train_acc = eval_model(model, trainloader, device_id=device,
                                     distributed=distributed)['acc']
        if log_dir is not None:
            logging.info(f'Quantized train accuracy: {quant_train_acc:.4f}')

        quant_test_acc = eval_model(model, testloader, device_id=device,
                                    distributed=distributed)['acc']
        if log_dir is not None:
            logging.info(f'Quantized test accuracy: {quant_test_acc:.4f}')
    else:
        ## FIXME: Implement distributed perturbations.
        raise NotImplementedError

    divergence = (prefix_message_len + divergence_gains + misc_extra_bits) * np.log(2)
    train_size = len(trainloader.dataset)
    if quant_train_acc < 0.5:
        err_bound = compute_catoni_bound(train_error=1. - quant_train_acc, divergence=divergence,
                                         sample_size=train_size)
    else:
        err_bound = pac_bayes_bound_opt(divergence=divergence, train_error=1. - quant_train_acc,
                                        n=train_size)
    return {
        'quant_train_err_100': (1 - quant_train_acc) * 100,
        'quant_test_err_100': (1 - quant_test_acc) * 100,
        'divergence_gains': divergence_gains,
        'err_bound_100': err_bound * 100,
        'misc_extra_bits': misc_extra_bits,
    }


def auto_eval(
    model,
    trainloader,
    testloader,
):
    options = [{'use_kmeans': 0, 'levels': 3}, {'use_kmeans': 0, 'levels': 5}]
    best_err = 200
    best_results = None
    best_option = None
    for option in options:
        results = evaluate_idmodel(model, trainloader, testloader, extra_bits=1, epochs=1, **option)
        if results['err_bound_100'] > best_err:
            best_err = results['err_bound_100']
            best_option = option
            best_results = results
    return {**best_results, **best_option}


def get_bounds(
    path,
    dataset,
    data_dir=None,
    subsample=False,
    rescale_posterior=1,
    encoding_type="arithmetic",
    levels=5,
    use_kmeans=0,
    epochs=10,
):
    """path: path to the saved pkl model"""
    rescale_posterior = bool(rescale_posterior)
    logging.getLogger().setLevel(logging.INFO)
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H%M_%S")
    string = 'bound_' + time_stamp + '_' + str(randint(1, int(1.e5)))
    filename = "logs/" + string + ".log"
    logs_exists = os.path.exists('./logs')
    if not logs_exists:
        os.mkdir('./logs')
    logging.basicConfig(filename=filename, level=logging.DEBUG)

    with FixedNumpySeed(0), FixedPytorchSeed(0):
        with open(path, 'rb') as f:
            model = dill.load(f)
    trainset, testset = get_dataset(dataset, root=get_data_dir(data_dir), extra_transform=None,
                                    aug=False)
    trainloader = DataLoader(trainset, batch_size=100, shuffle=False, num_workers=0,
                             pin_memory=False)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0, pin_memory=False)
    use_kmeans = bool(use_kmeans)
    quantize_kwargs = {'use_kmeans': use_kmeans, 'encoding_type': encoding_type, 'levels': levels}
    experiment_code = ''
    if use_kmeans:
        experiment_code += 'k'
    else:
        experiment_code += 'u'
    experiment_code += str(levels)
    experiment_code += str(encoding_type[0])
    results = evaluate_idmodel(model, trainloader, testloader, rescale_posterior=rescale_posterior,
                               subsample=subsample, epochs=epochs, **quantize_kwargs)
    results['code'] = experiment_code
    print(pd.Series(results))
    for key, value in results.items():
        logging.info(f'{key}: {value}')
    results = pd.DataFrame([results])
    results.to_csv(filename[:-4] + '.csv', index=False)

    return results
