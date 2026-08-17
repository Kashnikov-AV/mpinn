"""
Логика запуска экспериментов: обучение с Early Stopping и Grid Search.
"""
import time
import os
import pandas as pd
from functools import partial
from typing import Dict, List, Any, Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt

from .geom import Interval
from .pinn_core import PINN, FCNet
from .bc import dirichlet_bc, robin_bc
from mpinn.config import PhysicsParams, TrainConfig, get_activation, get_optimizer
from mpinn.plotting import show_plot, save_plot


def run_experiment(
    config: TrainConfig,
    phys: Optional[PhysicsParams] = None,
    pde_fn=None,
    exact_fn=None,
    bc_fns_override: Optional[List] = None,
) -> Tuple[Dict[str, Any], Dict[str, List[float]]]:
    """
    Запуск одного эксперимента с Early Stopping.
    
    Параметры:
    - config: конфигурация обучения
    - phys: физические параметры (если None, используются значения по умолчанию)
    - pde_fn: функция PDE (если None, используется line_1d)
    - exact_fn: функция точного решения для верификации
    - bc_fns_override: список функций граничных условий (если None, создаются автоматически)
    
    Возвращает:
    - metrics: словарь с метриками (mse, mape, и т.д.)
    - history: история обучения (losses по эпохам)
    """
    from .pde import line_1d
    from .analytic import line_1d_robin_exact
    
    if phys is None:
        phys = PhysicsParams()
    
    if pde_fn is None:
        pde_fn = line_1d
    
    if exact_fn is None:
        exact_fn = line_1d_robin_exact
    
    # Геометрия и точки коллокации
    geom = Interval(phys.x_left, phys.x_right)
    x_collocation = geom.generate_collocation(n_interior=config.num_points, method='random')
    
    # Создание модели
    act_fn = get_activation(config.activation_name)
    net = FCNet(
        din=1, 
        dmid=config.hidden_features, 
        dout=1, 
        num_layers=config.num_layers, 
        activation=act_fn, 
        rngs=nnx.Rngs(0)
    )
    
    optimizer = get_optimizer(config.opt_name, lr=config.lr)
    pinn = PINN(net, opt=optimizer, weights=list(config.weights))
    
    # Граничные условия
    if bc_fns_override is not None:
        bc_fns = bc_fns_override
    else:
        bc_fns = [
            partial(dirichlet_bc, x=phys.x_left, T=phys.T_left),
            partial(robin_bc, x=phys.x_right, alpha=phys.alpha_right, beta=phys.beta_right, h=phys.gamma_right)
        ]
    
    # Обучение с Early Stopping
    history, training_time = _train_with_early_stopping(
        pinn=pinn,
        x_collocation=x_collocation,
        pde_fn=pde_fn,
        bc_fns=bc_fns,
        phys=phys,
        max_epochs=config.max_epochs,
        patience=config.patience,
        min_delta=config.min_delta,
        monitor=config.monitor,
    )
    
    # Верификация
    x_test = jnp.linspace(phys.x_left, phys.x_right, 100).reshape(-1, 1)
    metrics, T_pred, T_exact = pinn.evaluate(
        x_test, exact_fn, phys, bc_names=['dirichlet', 'robin']
    )
    
    print(f'Эпохи: {len(history["steps"])}, Время: {training_time:.4f}c, MSE: {metrics["mse"]}')
    
    # Сохранение графика
    if config.save_img and config.image_path:
        os.makedirs(os.path.dirname(config.image_path), exist_ok=True)
        save_plot(x_test, T_pred, T_exact, phys, save_path=config.image_path)
    
    # Показ графика
    if config.show_plot:
        show_plot(x_test, T_pred, T_exact, phys)
    
    # Формирование результата
    result = {
        'bc_left': float(metrics['bc_left']),
        'bc_right': float(metrics['bc_right']),
        'training_time': f'{training_time:.4f}',
        'epochs_trained': len(history['steps']),
        'lr': config.lr,
        'activation_func': config.activation_name,
        'layers': config.num_layers,
        'neurons': config.hidden_features,
        'optimizer': config.opt_name,
        'collocation_points': config.num_points,
        'mape': float(metrics['mape']),
        'mae': float(metrics['mae']),
        'mse': float(metrics['mse']),
        'rmse': float(metrics['rmse']),
        'max_error': float(metrics['max_error']),
        'weights': str(config.weights),
    }
    
    return result, history


def _train_with_early_stopping(
    pinn: PINN,
    x_collocation: jnp.ndarray,
    pde_fn,
    bc_fns: List,
    phys: PhysicsParams,
    max_epochs: int,
    patience: int,
    min_delta: float,
    monitor: str = 'total_loss',
) -> Tuple[Dict[str, List[float]], float]:
    """
    Обучение модели с механизмом Early Stopping.
    
    Параметры:
    - pinn: объект PINN
    - x_collocation: точки коллокации
    - pde_fn: функция PDE
    - bc_fns: список функций граничных условий
    - phys: физические параметры
    - max_epochs: максимальное число эпох
    - patience: число эпох без улучшения до остановки
    - min_delta: минимальное изменение для учета как улучшения
    - monitor: имя метрики для мониторинга ('total_loss', 'pde', и т.д.)
    
    Возвращает:
    - history: история обучения
    - training_time: время обучения в секундах
    """
    import flax.nnx as nnx
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None
    
    history = {
        'steps': [],
        'total_loss': [],
        'pde': [],
        'bc_0': [],
        'bc_1': [],
    }
    
    start_time = time.time()
    
    for epoch in range(max_epochs):
        # Один шаг обучения
        losses = pinn.train_step(x_collocation, pde_fn, bc_fns, phys)
        
        current_loss = losses.get(monitor, losses['total_loss'])
        
        # Сохранение истории
        history['steps'].append(epoch)
        history['total_loss'].append(float(losses['total_loss']))
        history['pde'].append(float(losses['pde']))
        history['bc_0'].append(float(losses['bc_0']))
        history['bc_1'].append(float(losses['bc_1']))
        
        # Проверка улучшения
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            epochs_without_improvement = 0
            # Сохранение лучшего состояния модели
            best_state = nnx.state(pinn.net)
        else:
            epochs_without_improvement += 1
        
        # Ранняя остановка
        if epochs_without_improvement >= patience:
            print(f"Early stopping на эпохе {epoch}. Лучшая потеря: {best_loss:.6f}")
            break
    
    # Восстановление лучших весов
    if best_state is not None:
        nnx.update(pinn.net, best_state)
    
    training_time = time.time() - start_time
    
    return history, training_time


def run_grid_search(
    param_grid: Dict[str, List[Any]],
    base_config: Optional[TrainConfig] = None,
    phys: Optional[PhysicsParams] = None,
    csv_path: str = 'csv_results/1D_line_robin_results.csv',
    pde_fn=None,
    exact_fn=None,
) -> pd.DataFrame:
    """
    Grid Search: перебор комбинаций гиперпараметров.
    
    Параметры:
    - param_grid: словарь {имя_параметра: [список_значений]}
    - base_config: базовая конфигурация (если None, создается новая)
    - phys: физические параметры
    - csv_path: путь для сохранения результатов CSV
    - pde_fn, exact_fn: функции PDE и точного решения
    
    Возвращает:
    - DataFrame с результатами всех экспериментов
    """
    import itertools
    
    if base_config is None:
        base_config = TrainConfig()
    
    if phys is None:
        phys = PhysicsParams()
    
    # Генерация всех комбинаций параметров
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))
    
    results = []
    total_experiments = len(combinations)
    
    print(f"Запуск Grid Search: {total_experiments} комбинаций")
    
    for i, combo in enumerate(combinations, 1):
        print(f"Эксперимент {i}/{total_experiments}")
        
        # Создание конфига для текущей комбинации
        config_dict = {
            'hidden_features': base_config.hidden_features,
            'num_layers': base_config.num_layers,
            'activation_name': base_config.activation_name,
            'opt_name': base_config.opt_name,
            'lr': base_config.lr,
            'max_epochs': base_config.max_epochs,
            'patience': base_config.patience,
            'min_delta': base_config.min_delta,
            'num_points': base_config.num_points,
            'weights': base_config.weights,
            'save_img': False,
            'show_plot': False,
        }
        
        # Применение текущей комбинации параметров
        for key, value in zip(keys, combo):
            if key in config_dict:
                config_dict[key] = value
        
        config = TrainConfig(**config_dict)
        
        try:
            result, _ = run_experiment(
                config=config,
                phys=phys,
                pde_fn=pde_fn,
                exact_fn=exact_fn,
            )
            results.append(result)
        except Exception as e:
            print(f"Ошибка в эксперименте {i}: {e}")
            continue
    
    # Сохранение результатов
    if results:
        df_results = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = not os.path.exists(csv_path)
        df_results.to_csv(csv_path, mode='a', header=header, index=False)
        print(f"Результаты сохранены в {csv_path}")
        return df_results
    else:
        print("Нет успешных экспериментов для сохранения.")
        return pd.DataFrame()
