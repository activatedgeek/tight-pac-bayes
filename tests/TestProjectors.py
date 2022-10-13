import unittest
import numpy as np
import torch
from torch import nn
from pactl.nn.projectors import RoundedDoubleKronQR
from pactl.nn.projectors import find_locations_from_condition
from pactl.nn.projectors import flatten
from pactl.nn.projectors import unflatten_like
from pactl.nn.projectors import LazySTFiLMRDKronQR
from pactl.nn.projectors import find_all_batch_norm


class TestProjectors(unittest.TestCase):

    def test_find_all_batch_norm(self):
        net = BN_test()
        count = find_all_batch_norm(net)
        check = 64 * 2 + 32 * 2
        print('\nTEST: Count Batch Norm')
        diff = np.abs(check - count)
        print(f'Diff: {diff:1.3e}')
        self.assertTrue(expr=diff < 1.e-5)

    def test_LazySTFiLMRDKronQR(self):
        d = 100
        names = ['linear1.weight', 'bn1.weight', 'bn1.bias', 'linear2.bias', 'bn2.bias']
        params = [
            torch.randn(size=(30, 2)), torch.randn(size=(20,)),
            torch.randn(size=(15,)), torch.randn(size=(6,)),
            torch.randn(size=(19,)),
        ]
        d1 = 20 + 15 + 19
        d2 = d - d1
        D = len(flatten(params))
        bn = torch.arange(d1, dtype=torch.float32)
        ones = torch.ones(size=(d2,))
        P = LazySTFiLMRDKronQR(D, d, params, names)
        params_new = unflatten_like(P @ torch.concat([bn, ones]), params)
        params_check = unflatten_like(compute_STFiLMRDKronQR(names, params, d),  params)
        print('\nTEST: LazySTFiLMRDKronQR')
        for idx, p in enumerate(params_new):
            diff = np.linalg.norm(params_check[idx] - p)
            self.assertTrue(expr=diff < 1.e-5)

    def test_RoundedDoubleKronQR(self):
        mlp = MLP()
        D = sum([p.numel() for p in mlp.parameters()])
        names = [p[0] for p in mlp.named_parameters()]
        dd = 100
        P = RoundedDoubleKronQR(D, dd, params=mlp.parameters(), names=names)

        total = int(1.e1)
        results = np.zeros(total)
        means = []
        for i in range(total):
            z = torch.randn(dd)

            check = (P @ z).T @ (P @ z) / dd
            aux = z.T @ z / dd
            results[i] = torch.linalg.norm(aux - check)
        means.append(np.mean(results))
        diff = np.mean(results)
        print('\nTEST: Rounded Double Kron QR')
        print(f'Mean: {diff:1.3e}')
        self.assertTrue(expr=diff < 1.e-5)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = nn.Linear(10 * 5, 3)

    def forward(self, x):
        out = self.linear(x)
        return out


class BN_test(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32 * 32 * 3, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.linear2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.linear3 = nn.Linear(32, 10)

    def forward(self, x):
        out = nn.ReLU(self.bn1(self.linear1(nn.Flatten(x))))
        out = nn.ReLu(self.bn2(self.linear2(out)))
        return out


def compute_STFiLMRDKronQR(names, params, d):
    def condition_fn1(x): return x.find('bn') >= 0
    def condition_fn2(x): return not condition_fn1(x)
    flat_params = flatten(params)
    ids1 = find_locations_from_condition(names, params, condition_fn1)
    ids2 = find_locations_from_condition(names, params, condition_fn2)
    params_bn = flat_params[ids1]
    D, d1 = len(flat_params), len(params_bn)
    d2 = d - d1
    bn = torch.arange(d1, dtype=torch.float32)
    ones = torch.ones(size=(d2,))
    P_other = RoundedDoubleKronQR(D=D - d1, d=d2, params=params, names=names)
    check = torch.zeros(size=(D,))
    check[ids2] = P_other @ ones
    check[ids1] = bn
    return check


if __name__ == '__main__':
    unittest.main()
