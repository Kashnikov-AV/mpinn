"""Configuration for experiments: physics parameters, model and training hyperparameters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Callable, Any, Dict
import jax.numpy as jnp
import optax
from flax import nnx


@dataclass(frozen=True)
class PhysicsParams:
    _lambda: float = 1.0                     # теплопроводность
    source_fn: Callable | None = None        # объёмный источник
    T_min: float = 0.0                       # минимальная температура (для нормализации)
    T_max: float = 1000.0                    # максимальная температура (для нормализации)
    extra: Dict[str, Any] = field(default_factory=dict)   # произвольные параметры


@dataclass
class TrainConfig:
    """
    Конфигурация обучения: архитектура модели, оптимизатор, параметры обучения.
    Объединяет ModelConfig и TrainConfig.
    """

    # Архитектура модели
    hidden_features: int = 64
    num_layers: int = 2
    activation_name: str = "GELU"

    # Оптимизатор
    opt_name: str = "adam"
    lr: float = 0.01

    # Параметры обучения
    max_epochs: int = 3000
    patience: int = 200  # Число эпох без улучшения до остановки
    min_delta: float = 1e-6  # Минимальное изменение для учета как улучшения
    monitor: str = "total_loss"  # Метрика для мониторинга

    # Данные
    num_points: int = 100

    # Веса потерь [pde, bc] - один вес для PDE, один для всех BC
    weights: tuple[float, float] = field(default_factory=lambda: (1.0, 1.0))

    # Настройки вывода
    save_img: bool = False
    show_plot: bool = False
    image_path: str | None = None


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


# Параметры по умолчанию
DEFAULT_PHYSICS = {
    "x0": 0.0,
    "x1": 1.0,
    "T0": 300.0,
    "T_inf": 500.0,
    "_lambda": 1.0,
    "h": 10.0,
}

DEFAULT_TRAIN_CONFIG = {
    "hidden_features": 64,
    "num_layers": 2,
    "activation_name": "GELU",
    "opt_name": "adam",
    "lr": 0.01,
    "max_epochs": 3000,
    "patience": 200,
    "min_delta": 1e-6,
    "num_points": 100,
    "weights": (1.0, 1.0),
}
