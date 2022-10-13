import numpy as np
import torch
from torch import nn
from pactl.nn.projectors import RoundedKron
from pactl.nn.projectors import RoundedDoubleKron
from pactl.nn.projectors import LazyRandom
from pactl.nn.projectors import SparseOperator


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # self.linear = nn.Linear(5 * 5, 3)
        self.linear = nn.Linear(10 ** 5, 3)
        # self.linear = nn.Linear(10 ** 6, 3)

    def forward(self, x):
        out = self.linear(x)
        return out


mlp = MLP()

D = sum([p.numel() for p in mlp.parameters()])
names = [p[0] for p in mlp.named_parameters()]
# dds = [10, 20, 30, 40, 50, 100, 300]
dds = [10, 100, 500, 1000, 10 ** 4, 10 ** 5]
means = []

for dd in dds:
    # P = RoundedKron(D, dd, params=mlp.parameters(), names=names)
    # P = RoundedDoubleKron(D, dd, params=mlp.parameters(), names=names)
    # P = LazyRandom(D, dd, params=mlp.parameters(), names=names)
    P = RoundedDoubleKron(D, dd, params=mlp.parameters(), names=names)
    total = int(3.e1)
    results = np.zeros(total)
    for i in range(total):
        z = torch.randn(dd)

        check = (P @ z).T @ (P @ z) / dd
        aux = z.T @ z / dd
        print(aux)
        results[i] = torch.linalg.norm(aux - check)
        # print(f'Diff {results[i]:1.3e}')
    print(f'Mean: {np.mean(results):1.3e}')
    means.append(np.mean(results))

np.save('dds.npy', arr=np.array(dds))
np.save('means.npy', arr=np.array(means))
