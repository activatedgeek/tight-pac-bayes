from pathlib import Path

from experiments.train_gd import main
from pactl.logging import set_logging, finish_logging


def entrypoint(train_subset=.5, indices_path=None, **kwargs):
    ## First pretrain on a subset of the data.
    pretrain_kwargs = {
        **kwargs,
        'train_subset': train_subset,
        'indices_path': indices_path,
        'log_dir': set_logging(log_dir=kwargs.get('log_dir')),
        'intrinsic_dim': -1,
    }
    main(**pretrain_kwargs)

    ## Post-train with subset checkpoint.
    post_kwargs = {
        **kwargs,
        'train_subset': -train_subset,
        'indices_path': indices_path,
        'cfg_path': Path(pretrain_kwargs['log_dir']) / 'net.cfg.yml',
        'log_dir': pretrain_kwargs['log_dir'],
    }
    main(**post_kwargs)

    finish_logging()


if __name__ == '__main__':
    import fire
    fire.Fire(entrypoint)
