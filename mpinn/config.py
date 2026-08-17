"""
Конфигурация экспериментов: физические параметры, гиперпараметры модели и обучения.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import optax
import flax.nnx as nnx
import jax.numpy as jnp


@dataclass(frozen=True)
class PhysicsParams:
    """Физические параметры задачи."""
    x_left:  float = None
    x_right: float = None
    T_left:  float = None
    T_inf:   float = None
    _lambda: float = None
    h:       float = None

    @property
    def alpha_right(self):
        """Коэффициент при T в условии Робина: α = h"""
        return self.h

    @property
    def beta_right(self):
        """Коэффициент при dT/dx в условии Робина: β = λ"""
        return self._lambda

    @property
    def gamma_right(self):
        """Свободный член в условии Робина: γ = h · T_inf"""
        return self.h * self.T_inf


@dataclass
class TrainConfig:
    """
    Конфигурация обучения: архитектура модели, оптимизатор, параметры обучения.
    Объединяет ModelConfig и TrainConfig.
    """
    # Архитектура модели
    hidden_features: int = 64
    num_layers: int = 2
    activation_name: str = 'GELU'
    
    # Оптимизатор
    opt_name: str = 'adam'
    lr: float = 0.01
    
    # Параметры обучения
    max_epochs: int = 3000
    patience: int = 200  # Число эпох без улучшения до остановки
    min_delta: float = 1e-6  # Минимальное изменение для учета как улучшения
    monitor: str = 'total_loss'  # Метрика для мониторинга
    
    # Данные
    num_points: int = 100
    
    # Веса потерь [pde, bc_0, bc_1]
    weights: Tuple[float, float, float] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    
    # Настройки вывода
    save_img: bool = False
    show_plot: bool = False
    image_path: Optional[str] = None


def get_activation(name: str):
    """Фабрика функций активации."""
    mapping = {
        'tanh': nnx.tanh,
        'Swish': nnx.silu,
        'sin': jnp.sin,
        'GELU': nnx.gelu,
        'ReLU': nnx.relu,
        'Sigmoid': nnx.sigmoid,
    }
    return mapping.get(name, nnx.relu)


def get_optimizer(name: str, lr: float):
    """Фабрика оптимизаторов."""
    mapping = {
        'adam': optax.adam(lr),
        'sgd': optax.sgd(lr),
        'adagrad': optax.adagrad(lr),
        'rmsprop': optax.rmsprop(lr),
    }
    return mapping.get(name, optax.adam(lr))


# Параметры по умолчанию
DEFAULT_PHYSICS = {
    'x_left': 0.0,
    'x_right': 1.0,
    'T_left': 300.0,
    'T_inf': 500.0,
    '_lambda': 1.0,
    'h': 10.0,
}

DEFAULT_TRAIN_CONFIG = {
    'hidden_features': 64,
    'num_layers': 2,
    'activation_name': 'GELU',
    'opt_name': 'adam',
    'lr': 0.01,
    'max_epochs': 3000,
    'patience': 200,
    'min_delta': 1e-6,
    'num_points': 100,
    'weights': (1.0, 1.0, 1.0),
}
