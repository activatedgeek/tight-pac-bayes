from decimal import Decimal
from copy import deepcopy
import logging
from collections import Counter
from decimal import getcontext
import numpy as np
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
import scipy.stats
import torch
from torch.optim import SGD, Adam
import torch.nn as nn
from pactl.nn.projectors import _delchainattr
from pactl.prune_fns import get_pruned_vec


def finetune_prune_quantization(
    model,
    levels,
    device,
    train_loader,
    epochs,
    criterion,
    optimizer,
    lr,
    use_kmeans=False,
):
    vector = get_pruned_vec(model)
    vector = vector.detach().cpu().numpy()
    cluster_fn = get_random_symbols_and_codebook
    if use_kmeans:
        cluster_fn = get_kmeans_symbols_and_codebook
    _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
    has_zero = np.sum(centroids == 0.) > 0
    if not has_zero:
        aux = np.zeros(centroids.shape[0] + 1)
        aux[1:] = centroids
        centroids = aux
    centroids = torch.tensor(centroids, dtype=torch.float32)
    centroids = centroids.to(device)
    print(centroids)
    quantizer_fn = Quantize().apply
    qw = QuantizingWrapperPrune(model, quantizer=quantizer_fn, centroids=centroids)

    optim_params = [qw.centroids]
    if optimizer == "sgd":
        optimizer = SGD(
            optim_params,
            lr=lr,
        )
    elif optimizer == "adam":
        optimizer = Adam(optim_params, lr=lr)
    else:
        raise NotImplementedError

    run_sgd_prunned(train_loader, qw, criterion, optimizer, device=device, epochs=epochs)
    return qw


class QuantizingWrapperPrune(nn.Module):
    def __init__(self, net, quantizer, centroids):
        super().__init__()
        self.subspace_params = []
        self._forward_net = [net]
        self.named_params = list(net.named_parameters())
        for p_name, param in self.named_params:
            aux = nn.Parameter(deepcopy(param), requires_grad=True)
            self.subspace_params.append(aux)
            _delchainattr(net, p_name)
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=True)
        # self.centroids = torch.tensor(centroids, requires_grad=True, dtype=torch.float32)

    def forward(self, *args, **kwargs):
        for idx, (p_name, params) in enumerate(self.named_params):
            aux = self.subspace_params[idx]
            quant_params = self.quantizer(aux.reshape(-1), self.centroids)
            _setchainattr(
                self._forward_net[0], p_name, quant_params.reshape(*aux.shape)
            )
        return self._forward_net[0](*args, **kwargs)


def finetune_quantization(
    model,
    levels,
    device,
    train_loader,
    epochs,
    criterion,
    optimizer,
    lr,
    use_kmeans=False,
):
    vector = model.subspace_params.cpu().data.numpy()
    cluster_fn = get_random_symbols_and_codebook
    if use_kmeans:
        cluster_fn = get_kmeans_symbols_and_codebook
    _, centroids = cluster_fn(vector, levels=levels, codebook_dtype=np.float16)
    centroids = torch.tensor(centroids, dtype=torch.float32)
    centroids = centroids.to(device)
    quantizer_fn = Quantize().apply
    qw = QuantizingWrapper(model, quantizer=quantizer_fn, centroids=centroids)

    if optimizer == "sgd":
        optimizer = SGD(
            [qw.subspace_params, qw.centroids],
            lr=lr,
        )
    elif optimizer == "adam":
        optimizer = Adam([qw.subspace_params, qw.centroids], lr=lr)
    else:
        raise NotImplementedError

    run_sgd(
        train_loader,
        qw,
        criterion,
        optimizer,
        device=device,
        epochs=epochs,
    )
    return qw


def run_sgd_prunned(
    train_loader,
    net,
    criterion,
    optim,
    device=None,
    epochs=0,
):

    for e in tqdm(range(epochs)):
        net.train()
        logging.debug(f"centroids: {net.centroids}")
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device), Y.to(device)
            optim.zero_grad()
            f_hat = net(X)
            loss = criterion(f_hat, Y)
            loss.backward()
            net.centroids.grad[0] = 0.
            optim.step()

            if i % 100 == 0:
                metrics = {"epoch": e, "mini_loss": loss.detach().item()}
                logging.info(metrics, extra=dict(wandb=True, prefix="sgd/train"))


def run_sgd(
    train_loader,
    net,
    criterion,
    optim,
    device=None,
    epochs=0,
):

    for e in tqdm(range(epochs)):
        net.train()
        logging.debug(f"centroids: {net.centroids}")
        N_acc = 0
        N = len(train_loader.dataset)
        for i, (X, Y) in tqdm(enumerate(train_loader), leave=False):
            X, Y = X.to(device), Y.to(device)
            optim.zero_grad()
            f_hat = net(X)
            loss = criterion(f_hat, Y)
            loss.backward()
            optim.step()
            N_acc += (f_hat.argmax(dim=-1) == Y).sum()
            if i % 100 == 0:
                # print(f"Loss {loss}")
                # print(f"centroids: {net.centroids}")
                metrics = {"epoch": e, "mini_loss": loss.detach().item()}
                logging.info(metrics, extra=dict(wandb=True, prefix="sgd/train"))

        text = f"Loss: {loss:1.3e} | Acc: {N_acc.item() / N:2.3e}"
        print(text)
        print(f"centroids: {net.centroids}")


def create_assigment_matrix_prune(labels, num_clusters):
    assignments = torch.zeros(size=(num_clusters,) + labels.shape, device=labels.device)
    for k in range(num_clusters):
        assignments[k, labels == k] = 1.0
    return assignments


class QuantizingWrapper(nn.Module):
    def __init__(self, net, quantizer, centroids):
        super().__init__()
        # self.subspace_params = deepcopy(net.subspace_params)
        self.subspace_params = deepcopy(
            nn.Parameter(net.subspace_params, requires_grad=True)
        )
        _delchainattr(net, "subspace_params")

        self._forward_net = [net]
        self.quantizer = quantizer
        self.centroids = nn.Parameter(centroids, requires_grad=True)

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


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, centroids):
        vec = (centroids.unsqueeze(-2) - params.unsqueeze(-1)) ** 2.0
        mask = torch.min(vec, -1)[-1]
        ctx.assignment = create_assigment_matrix(mask, centroids.shape[0])
        quantized_params = centroids[mask]
        return quantized_params

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, ctx.assignment @ grad_output


def create_assigment_matrix(labels, num_clusters):
    size = (num_clusters, labels.shape[0])
    assignments = torch.zeros(size=size, device=labels.device)
    for k in range(num_clusters):
        assignments[k, labels == k] = 1.0
    return assignments


def quantize_vector(
    vec, levels=2**2 + 1, use_kmeans=False, encoding_type="arithmetic"
):
    codebook_dtype = np.float16
    if use_kmeans:
        symbols, codebook = get_kmeans_symbols_and_codebook(vec, levels, codebook_dtype)
    else:
        symbols, codebook = get_random_symbols_and_codebook(vec, levels, codebook_dtype)

    logging.info(f"KMeans: {use_kmeans}, Levels: {levels}, Algorithm: {encoding_type}")
    probabilities = np.array([np.mean(symbols == i) for i in range(levels)])
    logging.info(f"probs {probabilities}")

    if encoding_type == "arithmetic":
        _, coded_symbols_size = do_arithmetic_encoding(
            symbols, probabilities, levels
        )
    elif encoding_type == "huff":
        _, coded_symbols_size = do_huffman_encoding(symbols)
    else:
        NotImplementedError
    decoded_vec = np.zeros(shape=(len(vec)))
    for k in range(len(codebook)):
        decoded_vec[symbols == k] = codebook[k]

    message_len = get_message_len(coded_symbols_size, codebook, len(symbols))
    logging.info(f"Message Len: {message_len}")
    return decoded_vec, message_len


def get_random_symbols_and_codebook(vec, levels, codebook_dtype):
    largest = max(np.max(vec), np.abs(np.min(vec)))
    initvals = np.linspace(-largest - 1e-6, largest + 1e-6, levels + 1)
    assignments = np.digitize(vec, initvals) - 1
    centroids = []
    for i in range(levels):
        aux = vec[assignments == i]
        if len(aux) > 0:
            centroids.append(np.mean(aux))
        else:
            centroids.append(initvals[i])
    codebook = np.array(centroids, dtype=codebook_dtype)
    symbols = np.array(assignments)
    return symbols, codebook


def get_kmeans_symbols_and_codebook(vec, levels, codebook_dtype):
    kmeans = KMeans(n_clusters=levels).fit(vec.reshape(-1, 1))
    codebook = kmeans.cluster_centers_.astype(codebook_dtype)[:, 0]
    symbols = kmeans.labels_
    return symbols, codebook


def get_message_len(coded_symbols_size, codebook, max_count):
    codebook_bits_size = 16 if codebook.dtype == np.float16 else 32
    probability_bits = int(np.ceil(np.log2(max_count)) * len(codebook))
    codebook_bits = len(codebook) * codebook_bits_size
    summary = f"encoding {coded_symbols_size}, codebook {codebook_bits} probs {probability_bits}"
    logging.info(summary)
    message_len = coded_symbols_size + codebook_bits + probability_bits
    return message_len


def do_huffman_encoding(vec):
    vec_str = ""
    for i in range(len(vec)):
        vec_str += str(vec[i])
    freq = dict(Counter(vec_str))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)

    coded_symbols_len = 0
    for i in range(len(vec)):
        key = str(vec[i])
        key_size = len(encoding[key])
        coded_symbols_len += key_size
    return encoding, coded_symbols_len


def do_arithmetic_encoding(symbols, probabilities, levels):
    entropy_est = scipy.stats.entropy(probabilities, base=2)
    logging.info(f"Entropy: {entropy_est:.2f} bits")
    is_too_large_to_run = len(symbols) > int(1e4)
    if is_too_large_to_run:
        coded_symbols_size = np.ceil(len(symbols) * entropy_est) + 1.
    else:
        getcontext().prec = int(1.1 * np.log10(levels) * len(symbols))
        coded_symbols_size = len(encode(symbols, probabilities))
    return symbols, coded_symbols_size


def decimal2bits(decimal, bits_encoded):
    output_bits = []
    while len(output_bits) < bits_encoded:
        if decimal > Decimal(1) / Decimal(2):
            output_bits.append(1)
            decimal -= Decimal(1) / Decimal(2)
        else:
            output_bits.append(0)
        decimal *= Decimal(2)
    return output_bits


def bits2decimal(bits):
    val = Decimal(0)
    for i, bit in enumerate(bits):
        val += bit * Decimal(2) ** (-(i + 1))
    return val


def encode(sequence, probs):
    """Arithmetic coding of sequence of integers Seq: [a0,a1,a2,...]
    with probabilities: [c0,c1,c2,...]"""
    cumulative_probs = np.cumsum(probs)
    width = Decimal(1)
    message_value = Decimal(0)
    bits_encoded = 0
    for i, val in enumerate(sequence):
        bin_start = cumulative_probs[val - 1] if val > 0 else 0.0
        bin_size = probs[val]
        message_value = message_value + Decimal(bin_start) * width
        width = width * Decimal(bin_size)
        bits_encoded -= np.log2(bin_size)
    logging.info(f"arithmetic encoded bits {bits_encoded:.2f}")
    return decimal2bits(message_value + width / 2, np.ceil(bits_encoded) + 1)


def decode(bits, probs, N):
    """Arithmetic decoder which decodes bitstream using probabilities: [c0,c1,c2,...]"""
    message_val = bits2decimal(bits)
    cumulative_probs = np.cumsum(probs)
    width = Decimal(1)
    decoded_vals = []
    for i in range(N):
        bin_id = np.digitize(float(message_val), cumulative_probs)
        bin_start = cumulative_probs[bin_id - 1] if bin_id > 0 else 0.0
        bin_size = probs[bin_id]

        message_val = (message_val - Decimal(bin_start)) / Decimal(bin_size)
        width = width * Decimal(bin_size)
        decoded_vals.append(bin_id)
    return decoded_vals


class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def _setchainattr(obj, attr, value):
    attributes = attr.split(".")
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    # FIXME: not everything has to be a param
    # setattr(obj, attributes[-1], nn.Parameter(value))
    setattr(obj, attributes[-1], value)


def huffman_code_tree(node, binString=""):
    """
    Function to find Huffman Code
    """
    if type(node) is str:
        return {node: binString}
    (l, r) = node.children()
    d = dict()
    d.update(huffman_code_tree(l, binString + "0"))
    d.update(huffman_code_tree(r, binString + "1"))
    return d


def make_tree(nodes):
    """
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    """
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]


if __name__ == "__main__":
    vec = np.array([1, 2, 0, 0, 3, 3, 3, 2, 2, 0, 2, 0, 2, 0, 2])
    encoding = do_huffman_encoding(vec)
    for i in encoding:
        print(f"{i} : {encoding[i]}")
