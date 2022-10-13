import os
import torch


def maybe_launch_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = 0
    device_id = 0
    if world_size > 1:    
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        assert world_size == torch.distributed.get_world_size()

        rank = torch.distributed.get_rank()
        device_id = int(os.environ.get('LOCAL_RANK'))

    return world_size, rank, device_id


class DistributedValue:
    def __init__(self, init_val):
        self.val = init_val

    def __iadd__(self, v):
        self.val += v
        return self

    def resolve(self):
        torch.distributed.all_reduce(self.val, torch.distributed.ReduceOp.SUM, async_op=False)
        return self.val
