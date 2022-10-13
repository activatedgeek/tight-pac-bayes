import unittest
import torch
from torch import nn
import torch.nn.functional as F
from pactl.prune_fns import prune_params
from pactl.prune_fns import get_pruned_vec
from pactl.prune_fns import make_prunning_permanent
from pactl.prune_fns import flatten


class TestPrune(unittest.TestCase):
    def test_make_prunning_permanent(self):
        nets = [NN(), LeNet()]
        print("\nTEST: making prunning persistent")
        for model in nets:
            prune_params(model, amount=0.1)
            aux1 = [n for n, _ in model.named_parameters()]
            print(aux1)
            make_prunning_permanent(model)
            aux2 = [n for n, _ in model.named_parameters()]
            print(aux2)
            count = count_prunning_masks(aux2)
            self.assertTrue(count == 0)
            self.assertTrue(count < count_prunning_masks(aux1))

    def test_pruning(self):
        test_tol = 1.0e-0
        # x = torch.ones(size=(10, 1, 28, 28))
        amount = 0.9
        model = LeNet()
        prune_params(model, amount=amount)
        vec = flatten(list(model.parameters()))
        aux = len(vec) * 0.9
        initial_mask = vec == 0.0
        quantized_vec = get_pruned_vec(model)
        mask = quantized_vec == 0.0
        total_zeros = torch.sum(mask)
        print("\nTEST: Pruning")
        print(f"Initial numbers of zeros: {torch.sum(initial_mask)}")
        print(f"Final numbers of zeros:   {torch.sum(mask):,}")
        diff = torch.abs(total_zeros - aux)
        self.assertTrue(expr=diff < test_tol)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


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


def count_prunning_masks(names):
    count = 0
    for name in names:
        if name.find("orig") >= 0:
            count += 1
    return count


if __name__ == "__main__":
    unittest.main()
