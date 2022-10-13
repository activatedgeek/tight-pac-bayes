# PAC-TL

## Setup

```shell
conda env create -f environment.yml -n pactl
```

Setup the `pactl` package.

```shell
pip install -e .
```

## Experiment

The simplest example for training CIFAR-10 as

```shell
python experiments/train.py --dataset=cifar10 \
                            --model-name=resnet18k \
                            --epochs=200 \
                            --lr=.1
```

To transfer from a trained model checkpoint, use

```shell
python experiments/train.py --dataset=cifar10 \
                            --cfg-path=<cfg_path> \
                            --epochs=200 \
                            --lr=.1
```

For distributed multi-GPU training on a single machine, replace `python` with `torchrun --nproc_per_node=<num_gpus_per_node>`.

To compute PAC-Bayes bound from a checkpoint, use

```shell
python experiments/compute_bound.py --dataset=mnist \
                                    --prenet-cfg-path=<cfg_path> \
                                    --levels=5 \
                                    --use_kmeans=1
```
