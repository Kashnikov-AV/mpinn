"""Configuration for experiments: physics parameters, model and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any, Dict, Optional

import jax.numpy as jnp
import optax
from flax import nnx


@dataclass(frozen=True)
class PhysicsParams:
    """
    Физические параметры задачи.
    Все основные параметры должны быть переданы явно.
    """
    _lambda: float                     # теплопроводность
    T_min: float                       # минимальная температура (для нормализации)
    T_max: float                       # максимальная температура (для нормализации)
    source_fn: Optional[Callable] = None   # объёмный источник (опционально)


@dataclass
class TrainConfig:
    # Обязательные поля (без дефолтов) — идут первыми
    hidden_features: int
    num_layers: int
    activation_name: str
    opt_name: str
    lr: float
    epochs: int
    num_points: int
    n_test_points: int

    # Необязательные поля
    log_interval: int = 100
    weights: tuple[float, float] = (1.0, 1.0)


def get_activation(name: str):
    """Фабрика функций активации."""
    mapping = {
        "tanh": nnx.tanh,
        "Swish": nnx.silu,
        "sin": jnp.sin,
        "GELU": nnx.gelu,
        "ReLU": nnx.relu,
        "Sigmoid": nnx.sigmoid,
    }
    return mapping.get(name, nnx.relu)


def get_optimizer(name: str, lr: float):
    """Фабрика оптимизаторов."""
    mapping = {
        "adam": optax.adam(lr),
        "sgd": optax.sgd(lr),
        "adagrad": optax.adagrad(lr),
        "rmsprop": optax.rmsprop(lr),
    }
    return mapping.get(name, optax.adam(lr))