import torch
import torch.nn.utils.prune as prune


def make_prunning_permanent(net):
    instances = (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)
    for module in net.modules():
        if isinstance(module, instances):
            prune.remove(module, "weight")


def prune_params(net, amount=0.9):
    params_to_prune = get_params_to_prune(net)
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

def recover_prune_mask(net):
    params_to_prune = get_params_to_prune(net)
    prune.global_unstructured(
        params_to_prune,
        pruning_method=RecoverPruneMask,
    )

class RecoverPruneMask(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = torch.zeros_like(t)
        mask[t.nonzero()] = 1.
        return mask


def get_params_to_prune(net):
    instances = (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)
    params_to_prune = [
        (module, "weight") for module in net.modules() if isinstance(module, instances)
    ]
    # params_to_prune += [(module, 'bias')
    #                     for module in net.modules() if isinstance(module, instances)]
    return params_to_prune


def get_pruned_vec(net):
    vecs = get_pruned_params(net)
    vec = flatten(vecs)
    return vec


def get_pruned_params(net):
    instances = (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d)
    params = []
    for module in net.modules():
        if isinstance(module, instances):
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    return params


def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)
