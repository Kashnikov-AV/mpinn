"""
Модуль экспериментов для 1D задач теплопроводности.
Включает конфигурацию, запуск обучения и перебор параметров.
"""

from experiments.config import PhysicsParams, TrainConfig, get_activation, get_optimizer, DEFAULT_PHYSICS, DEFAULT_TRAIN_CONFIG
from experiments.plotting import show_plot, save_plot
from experiments.runner import run_experiment, run_grid_search

__all__ = [
    'PhysicsParams',
    'TrainConfig',
    'get_activation',
    'get_optimizer',
    'DEFAULT_PHYSICS',
    'DEFAULT_TRAIN_CONFIG',
    'show_plot',
    'save_plot',
    'run_experiment',
    'run_grid_search',
]
