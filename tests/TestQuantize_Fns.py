import unittest
import sys
import numpy as np
from decimal import getcontext
import scipy.stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MSELoss
from pactl.prune_fns import prune_params
from pactl.bounds.quantize_fns import quantize_vector
from pactl.bounds.quantize_fns import Quantize
from pactl.bounds.quantize_fns import do_huffman_encoding
from pactl.bounds.quantize_fns import encode
from pactl.bounds.quantize_fns import decode
from pactl.bounds.quantize_fns import finetune_prune_quantization
from pactl.bounds.quantize_fns import QuantizingWrapperPrune
from experiments.quantize_pruned_model import update_params


class TestQuantize_Fns(unittest.TestCase):

    def test_centroids_grads(self):
        loss = MSELoss()
        nn = NN()
        prune_params(nn, amount=0.5)
        batch_size, dim_n = 10, 2
        x = torch.randn(size=(batch_size, dim_n))
        y = torch.ones(size=(batch_size, 1))
        centroids = torch.tensor([0., -0.3, 0.2])

        quantizer_fn = Quantize().apply
        qw = QuantizingWrapperPrune(nn, quantizer_fn, centroids=centroids)
        logits = qw(x)
        output = loss(logits, y)
        output.backward()
        aux = [param for param in qw.parameters()]
        self.assertTrue(expr=aux[0].grad is not None)

    def test_prunning_evaluation(self):
        nn = NN()
        prune_params(nn, amount=0.5)
        scale = 0.05
        orig_weights = [p.clone() for p in nn.parameters()]
        update_params(nn, orig_weights, scale)
        new_weights = [p.clone() for p in nn.parameters()]
        diff = torch.linalg.norm(orig_weights[0] - new_weights[0])
        print("TEST: add noise on params")
        self.assertTrue(expr=diff > 1e-6)

    def test_quant_prune(self):
        test_tol = 1.e-6
        criterion = torch.nn.CrossEntropyLoss()
        epochs, optimizer, lr = 2, "sgd", 1.e-3
        levels, batch_n, device, use_kmeans = 5, 10, None, False
        x, y = torch.ones(size=(1, 28, 28)), 0
        data = [(x, y) for _ in range(batch_n)]
        train_loader = torch.utils.data.DataLoader(data, batch_size=2)
        model = LeNet()
        prune_params(model, amount=0.9)
        manual = [np.sum(np.abs(p.detach().numpy())) for p in model.parameters()]
        manual = np.sum(manual)
        qw = finetune_prune_quantization(
            model=model,
            levels=levels,
            device=device,
            train_loader=train_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            use_kmeans=use_kmeans)
        check = [np.sum(np.abs(p.detach().numpy())) for p in qw.subspace_params]
        check = np.sum([np.sum(np.abs(p.detach().numpy())) for p in qw.subspace_params])
        print("\nTest: Quantize Pruned Model Step")
        print(f"\nManual: {manual:1.3f}")
        print(f"\nCheck: {check:1.3f}")
        diff = np.abs(manual - check)
        self.assertTrue(expr=diff > test_tol)

    def test_huffman_encoding(self):
        test_tol = 1.e-10
        vec = np.array([1, 2, 0, 0, 3, 3, 3, 2, 2, 0, 2, 0, 2, 0, 2])
        encoded, coded_symbols_len = do_huffman_encoding(vec)
        manual = {'2': '0', '1': '100', '3': '101', '0': '11'}
        print('\nTEST: Huffman Encoding')
        for k, v in manual.items():
            self.assertTrue(expr=manual[k] == encoded[k])
        self.assertTrue(expr=np.abs(coded_symbols_len - 28) < test_tol)

    def test_kmeans_quant(self):
        test_tol = 1.e-10
        vec = torch.tensor([1., 1.1, 0.9, 3., 3.5])
        # centroids, labels = get_centroids_and_labels(vec, n_clusters=2)
        centroids, xx = quantize_vector(vec, levels=2, use_kmeans=True)
        manual_c = torch.tensor([1., 1., 1., 3.25, 3.25])
        # manual_l = np.array([0, 0, 0, 1, 1])
        print('\nTEST: K-means quantization')
        diff = np.linalg.norm(manual_c - centroids)
        # diff += np.linalg.norm(manual_l - labels)
        self.assertTrue(expr=diff < test_tol)

    def test_arithmetic(self):
        levels = 32
        x = np.linspace(-1, 1, levels)
        probabilities = np.exp(-5*x**2)
        probabilities = probabilities/probabilities.sum()
        qids = np.random.choice(levels, p=probabilities, size=(1000,))
        # IMPORTANT. Set decimal precision
        getcontext().prec = int((8/3) * sys.getsizeof(qids))
        encoded = encode(qids, probabilities)
        print(
            f"Message length: {len(encoded)} bits, " +
            f"avg {len(encoded)/len(qids)} bpd, " +
            f"entropy {scipy.stats.entropy(probabilities,base=2)}")
        decoded = decode(encoded, probabilities, len(qids))
        self.assertTrue(expr=np.linalg.norm(np.array(decoded) - qids) < 1.e-6)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2, out_features=4),
            nn.ReLU(),
            nn.Linear(in_features=4, out_features=6),
            nn.ReLU(),
            nn.Linear(in_features=6, out_features=1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    unittest.main()
