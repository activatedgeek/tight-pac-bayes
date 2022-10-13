import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
__all__ = ['IntrinsicDenseNet','LazyIntrinsicDenseNet']


def _getchainattr(obj, attr):
    attributes = attr.split('.')
    for a in attributes:
        obj = getattr(obj, a)
    return obj

def _delchainattr(obj, attr):
    attributes = attr.split('.')
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    try:
        delattr(obj, attributes[-1])
    except AttributeError:
        print(obj)
        raise

def _setchainattr(obj, attr, value):
    attributes = attr.split('.')
    for a in attributes[:-1]:
        obj = getattr(obj, a)
    setattr(obj, attributes[-1], value)

class IntrinsicDenseNet(nn.Module):
    """ Intrinsic dimensionality training warpper of a given model.
        Will use a dense random projection of dimension DIMENSION
        of the parameters of net that require grad.
    """
    def __init__(self, net, dimension=5):
        super().__init__()

        self.d = dimension

        ## 
        # Exclude from .parameters() generator.
        # https://discuss.pytorch.org/t/how-to-exclude-parameters-from-model/6151
        #
        self._orig_net = [net]
        self._init_net = [deepcopy(net)]

        self._proj_name = []
        self._proj_mat = nn.ParameterList()
        _proj_norm = 0.
        D = 0
        for orig_name, orig_p in self._init_net[0].named_parameters():
            if not orig_p.requires_grad:
                self._proj_name.append(None)
                self._proj_mat.append(None)
                continue
            ## Delete old parameter leaf.
            self._proj_name.append(orig_name)
            _delchainattr(net, orig_name)

            _proj_block = torch.randn(self.d, *orig_p.shape)
            self._proj_mat.append(nn.Parameter(_proj_block).requires_grad_(False))
            flat_proj_block = _proj_block.view(self.d, -1)
            _proj_norm += flat_proj_block.pow(2).sum(dim=-1)
            D += flat_proj_block.shape[-1]
        self._init_net = [self._init_net[0].requires_grad_(False)]
        ## Normalize projection matrix.
        _proj_norm = (_proj_norm + 1e-8).sqrt()
        for p in self._proj_mat:
            if p is None:
                continue
            p /= _proj_norm.view(-1, *([1] * len(p.shape[1:])))

        ## New parameter leaf.
        self.subspace_params = nn.Parameter(torch.zeros(self.d))
        print(f"Intrinsic dimensionality with projection matrix: {D} x {self.d}")

    def to(self, *args, **kwargs):
        self._orig_net[0].to(*args, **kwargs)
        self._init_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        #print(len(self._proj_name),len(list(self._proj_mat.parameters())),len(list(self._init_net[0].parameters())))
        for p_name, init, proj_mat in zip(self._proj_name,
                                            self._init_net[0].parameters(), self._proj_mat):
            if p_name is None:
                continue
            p = init + (proj_mat * self.subspace_params.view(-1, *([1] * len(init.shape)))).sum(dim=0)

            ## Replace original leaf.
            _setchainattr(self._orig_net[0], p_name, p)

        return self._orig_net[0](*args, **kwargs)


class FixedPytorchSeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.pt_rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, *args):
        torch.random.set_rng_state(self.pt_rng_state)


class LazyIntrinsicDenseNet(nn.Module):
    """ Intrinsic dimensionality training warpper of a given model.
        Will use a dense random projection of dimension DIMENSION
        of the parameters of net that require grad.
        Recomputes the projection matrix on the fly
    """
    def __init__(self, net, dimension=5,seed=0):
        super().__init__()

        self.d = dimension
        self.seed=seed
        ## 
        # Exclude from .parameters() generator.
        # https://discuss.pytorch.org/t/how-to-exclude-parameters-from-model/6151
        #
        self._orig_net = [net]
        self._init_net = [deepcopy(net)]

        self._proj_name = []
        _proj_norm = 0.
        self.D = 0
        for orig_name, orig_p in self._init_net[0].named_parameters():
            if not orig_p.requires_grad:
                self._proj_name.append(None)
                continue
            ## Delete old parameter leaf.
            self._proj_name.append(orig_name)
            _delchainattr(net, orig_name)
            self.D += np.prod(orig_p.shape)
        self._init_net = [self._init_net[0].requires_grad_(False)]
        
        ## New parameter leaf.
        self.subspace_params = nn.Parameter(torch.zeros(self.d))
        print(f"Intrinsic dimensionality with projection matrix: {self.D} x {self.d}")

    def to(self, *args, **kwargs):
        self._orig_net[0].to(*args, **kwargs)
        self._init_net[0].to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        shapes = [(self.d, np.prod(init.shape)) for name, init in zip(self._proj_name,self._init_net[0].parameters()) if name is not None]
        projected_params = RandomParamProjection.apply(self.subspace_params, shapes)
        proj_param_iter = iter(projected_params)
        for p_name, init in zip(self._proj_name,self._init_net[0].parameters()):
            if p_name is None:
                continue
            #print(p_name,init.shape,proj_params.shape)
            p = init + (next(proj_param_iter)).reshape(*init.shape)/np.sqrt(self.D)
            ## Replace original leaf.
            _setchainattr(self._orig_net[0], p_name, p)
        return self._orig_net[0](*args, **kwargs)

class RandomParamProjection(torch.autograd.Function):
    #CHUNK_MAX=2e8
    @staticmethod
    def forward(ctx, subspace_params,shapes):
        #ctx.save_for_backward(shapes)
        ctx.shapes = shapes
        paramlist = []
        with FixedPytorchSeed(0):
            for s in shapes:
                paramlist.append(subspace_params@torch.randn(*s,device=subspace_params.device))
        out = tuple(paramlist)
        #print(out)
        return out

    @staticmethod
    def backward(ctx, *grad_output):
        #print(grad_output)
        shapes = ctx.shapes
        grad_in = 0.
        with FixedPytorchSeed(0):
            for s,gp in zip(shapes,grad_output):
                # if s is None:
                #     continue
                grad_in += gp@torch.randn(*s,device=gp.device).T
        return grad_in,None