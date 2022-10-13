from pathlib import Path
from uuid import uuid4
import logging
from logging.config import dictConfig
import os
import time
import wandb
import platform


class WandBNop:
    config = dict()
    class _Summary:
        summary = dict()
    run = _Summary()
    def nop(self, *_, **__): pass
    def __getattr__(self, _): return self.nop


## Overload wandb on Windows. See issue https://github.com/wandb/client/issues/1370.
if platform.system() == 'Windows':
    wandb = WandBNop()


class MetricsFilter(logging.Filter):
    def __init__(self, extra_key='metrics', invert=False):
        super().__init__()
        self.extra_key = extra_key
        self.invert = invert

    def filter(self, record):
        should_pass = hasattr(record, self.extra_key) and getattr(record, self.extra_key)
        if self.invert:
            should_pass = not should_pass
        return should_pass


class MetricsFileHandler(logging.FileHandler):
    def emit(self, record):
        if hasattr(record, 'prefix'):
            record.msg = {f'{record.prefix}/{k}': v for k, v in record.msg.items()}
        record.msg['timestamp_ns'] = time.time_ns()
        return super().emit(record)


class WnBHandler(logging.Handler):
    '''Listen for W&B logs.

    Default Usage:
    ```
    logging.log(metrics_dict, extra=dict(metrics=True, prefix='train'))
    ```

    `metrics_dict` (optionally prefixed) is directly consumed by `wandb.log`.
    '''
    def emit(self, record):
        metrics = record.msg
        if hasattr(record, 'prefix'):
            metrics = {f'{record.prefix}/{k}': v for k, v in metrics.items()}
        wandb.log(metrics)


def get_log_dir(log_dir=None):
    if log_dir is not None:
        return Path(log_dir)

    root_dir = None
    if not isinstance(wandb, WandBNop):
        root_dir = Path(wandb.run.dir) / '..'
    else:
        root_dir = Path(os.environ.get('LOGDIR', Path.cwd() / '.log')) / f'run-{str(uuid4())[:8]}'

    log_dir = Path(str((root_dir / 'files').resolve()))
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir


def set_logging(metrics_extra_key='wandb', log_dir=None):
    wandb.init(mode=os.environ.get('WANDB_MODE', default='offline'))

    log_dir = get_log_dir(log_dir=log_dir)

    _CONFIG = {
        'version': 1,
        'formatters': {
            'console': {
                'format': '[%(asctime)s] (%(funcName)s:%(levelname)s) %(message)s',
            },
        },
        'filters': {
            'metrics': {
                '()': MetricsFilter,
                'extra_key': metrics_extra_key,
            },
            'nometrics': {
                '()': MetricsFilter,
                'extra_key': metrics_extra_key,
                'invert': True,
            },
        },
        'handlers': {
            'stdout': {
                '()': logging.StreamHandler,
                'formatter': 'console',
                'stream': 'ext://sys.stdout',
                'filters': ['nometrics'],
            },
            ## For using plain file logger.
            # 'metrics_file': {
            #     '()': MetricsFileHandler,
            #     'filename': str(Path(log_dir) / 'metrics.log'),
            #     'filters': ['metrics'],
            # },
            'metrics_file': {
                '()': WnBHandler,
                'filters': ['metrics'],
            },
        },
        'loggers': {
            '': {
                'handlers': ['stdout', 'metrics_file'],
                'level': os.environ.get('LOGLEVEL', 'INFO'),
            },
        },
    }

    dictConfig(_CONFIG)

    logging.info(f'Files stored in "{log_dir}".')

    return log_dir


def finish_logging():
    wandb.finish()
