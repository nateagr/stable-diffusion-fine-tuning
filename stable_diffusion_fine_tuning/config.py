from typing import NamedTuple
from enum import Enum


class Precision(Enum):
    FP32 = 1
    FP16 = 2
    AMP = 3


class AdamConfig(NamedTuple):
    lr = 2e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    weight_decay = 1e-5


class DatasetConfig(NamedTuple):
    path = "/home/username/"
    training_ds_ratio = 0.75
    train_batch_size = 1
    test_batch_size = 1


class GradientConfig(NamedTuple):
    grad_acc: int = 1
    gradient_checkpointing = False
    enable_grad = lambda _, x: "norm" in x or "bias" in x or "emb" in x or "attn" in x
