# Tight PAC-Bayes Compression Bounds

[![](https://img.shields.io/badge/arXiv-2211.xxxxx-red)]() [![](https://img.shields.io/badge/NeurIPS-2022-green)]()

This repository hosts the code for [PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization]() by [Sanae Lotfi*](https://sanaelotfi.github.io), [Marc Finzi*](https://mfinzi.github.io), [Sanyam Kapoor*](https://sanyamkapoor.com), [Andres Potapczynski*](https://www.andpotap.com), [Micah Goldblum](https://goldblum.github.io), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

## Setup

```shell
conda env create -f environment.yml -n pactl
```

Setup the `pactl` package.

```shell
pip install -e .
```

## Usage

We use [Fire](https://google.github.io/python-fire/guide/) for CLI parsing.

### Training Intrinsic Dimensionality Models


```shell
python experiments/train.py --dataset=cifar10 \
                            --model-name=resnet18k \
                            --base-width=64 \
                            --optimizer=adam \
                            --epochs=500 \
                            --lr=1e-3 \
                            --intrinsic_dim=1000 \
                            --intrinsic_mode=rdkronqr \
                            --seed=137
```

All arguments in the `main` method of [experiments/train.py](./experiments/train.py)
are valid CLI arguments. The most imporant ones are noted here:

* `--seed`: Setting the seed is important so that any subsequent runs using the checkpoint can reconstruct the same random parameter projection matrices used during training.
* `--data_dir`: Parent path to directory containing root directory of the dataset.
* `--dataset`: Dataset name. See [data.py](./pactl/data.py) for list of dataset strings.
* `--intrinsic_dim`: Dimension of the training subspace of parameters.
* `--intrinsic_mode`: Method used to generate (sparse) random projection matrices. See `create_intrinsic_model` method in [projectors.py](./pactl/nn/projectors.py) for a list of valid modes.

#### Distributed Training

Distributed training is helpful for large datasets like Imagenet to spread computation over multiple GPUs. 
We rely on [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

To use multiple GPUs on a single node, we need:
* GPU visibility flags appropriately via `CUDA_VISIBLE_DEVICES`.
* Specify the number `x` of GPUs made visible via `--nproc_per_node=<x>`
* Specify a random port `yyyy` on the host for inter-process communication via `--rdzv_endpoint=localhost:yyyy`.

For the same run as above, we simply replace `python` with `torchrun` as:
```shell
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:9999 experiments/train.py ...
```
All remaining CLI arguments remain unchanged.

### Transfer Learning using Existing Checkpoints

The key argument needed for transfer is the path to the configuration file named `net.cfg.yml` of the pretrained network. 

```shell
python experiments/train.py --dataset=fmnist \
                            --optimizer=adam \
                            --epochs=500 \
                            --lr=1e-3 \
                            --intrinsic_dim=1000 \
                            --intrinsic_mode=rdkronqr \
                            --prenet_cfg_path=<path/to/net.cfg.yml> \
                            --seed=137 \
                            --transfer
```

In addition to earlier arguments, there is only one new key argument:
* `--prenet_cfg_path`: Path to `net.cfg.yml` configuration file of the pretrained network. This path is logged during the train command specified previously.

### Training for Data-Dependent Bounds

Data-dependent bounds first require pre-training on a fixed subset of training data and then training
an intrinsic dimensionality model on the remainder of the subset.

For such training, we use the following command:
```shell
python experiments/train_dd_priors.py --dataset=cifar10 \
                                      ...
                                      --indices_path=<path/to/index/list> \
                                      --train-subset=0.1 \
                                      --seed=137
```

The key new arguments here in addition to the ones seen previously are:
* `--indices-path`: A fixed permutation of indices as a numpy list equal to the length of the dataset. If not specified, a random permutation is generated every time and the results may not be reproducible. See [dataset_permutations.ipynb](./notebooks/dataset_permutations.ipynb) to see an example of how to generate such a file.
* `--train-subset`: A fractional subset of the training data to use. If a negative fraction, then the complement is used.

### Computing our Adaptive Compression Bounds

Once we have the checkpoints of intrinsic-dimensionality models, the bound can be computed using:

```shell
python experiments/compute_bound.py --dataset=mnist \
                                    --misc-extra-bits=7 \
                                    --quant-epochs=30 \
                                    --levels=50 \
                                    --lr=0.0001 \
                                    --prenet_cfg_path=<path/to/net.cfg.yml> \
                                    --use_kmeans=True
```

The key arguments here are:
* `--misc-extra-bit`: Penalty for hyper-parameter optimization during bound computation, equals the bits required to encode all hyper-parameter configurations.
* `--levels`: Number of quantization levels.
* `--quant-epochs`: Number of epochs used for fine-tuning of quantization levels.
* `--lr`: Learning rate used for fine-tuning of quantization levels.
* `--user_kmeans`: When true, uses kMeans clustering for initialization of quantization levels. Otherwise, random initialization is used.

## LICENSE

Apache 2.0
