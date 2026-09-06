"""
Запуск экспериментов PINN с Grid Search.
Поддерживает 1D, 2D, 3D.
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pandas as pd
from flax import nnx

from mpinn.config import PhysicsParams, TrainConfig, get_activation, get_optimizer
from mpinn.geom import GeometryBase
from mpinn.pinn_core import PINN, FCNet, NormalizedNet


def run_experiment(
    config: TrainConfig,
    geom: GeometryBase,
    phys: PhysicsParams,
    pde_fn: Callable,
    exact_fn: Callable,
    bc_configs: List[Dict[str, Any]],
    bc_info: Optional[Dict[str, str]] = None,
    x_test: Optional[jnp.ndarray] = None,
    return_model: bool = False,
) -> Union[Tuple[Dict[str, Any], Dict[str, List[float]]],
           Tuple[Dict[str, Any], Dict[str, List[float]], PINN]]:
    """
    Запускает один эксперимент PINN с нормализацией.

    Args:
        config: конфигурация обучения.
        geom: геометрия.
        phys: физические параметры.
        pde_fn: функция PDE.
        exact_fn: функция точного решения (может быть None).
        bc_configs: список конфигураций ГУ.
        bc_info: информация о ГУ для логирования.
        x_test: тестовые точки (если None – генерируются).
        return_model: если True, возвращает также обученную модель PINN.

    Returns:
        Если return_model=False: (result, history)
        Если return_model=True: (result, history, pinn)
    """
    x_collocation = geom.sample_interior(
        n_points=config.num_points,
        method="random",
        rng=jax.random.PRNGKey(0),
    )

    if hasattr(geom, "bounds"):
        x_min, x_max = geom.bounds
        if isinstance(x_min, jnp.ndarray) and x_min.ndim == 1:
            x_min = x_min[0]
            x_max = x_max[0]
    elif hasattr(geom, "x0") and hasattr(geom, "x1"):
        x_min, x_max = geom.x0, geom.x1
    else:
        raise ValueError("Не удалось определить границы координат.")

    if not hasattr(phys, "T_min") or not hasattr(phys, "T_max"):
        raise ValueError("PhysicsParams должен содержать T_min и T_max.")

    act_fn = get_activation(config.activation_name)
    base_net = FCNet(
        din=geom.dim,
        dmid=config.hidden_features,
        dout=1,
        num_layers=config.num_layers,
        activation=act_fn,
        rngs=nnx.Rngs(0),
    )
    net = NormalizedNet(base_net, x_min, x_max, phys.T_min, phys.T_max)

    optimizer = get_optimizer(config.opt_name, lr=config.lr)
    pinn = PINN(
        net=net,
        opt=optimizer,
        weights=list(config.weights),
        phys=phys,
        pde_fn=pde_fn,
        bc_configs=bc_configs
    )

    history, training_time = pinn.fit(
        x_collocation=x_collocation,
        epochs=config.epochs,
        log_interval=config.log_interval
    )

    if x_test is None:
        x_test = generate_test_points(geom, config.n_test_points)

    metrics, T_pred, T_exact = pinn.evaluate(
        x_test, exact_fn, phys, bc_info=bc_info
    )

    result = {
        "training_time": f"{training_time:.4f}",
        "epochs_trained": config.epochs,
        "lr": config.lr,
        "activation_func": config.activation_name,
        "layers": config.num_layers,
        "neurons": config.hidden_features,
        "optimizer": config.opt_name,
        "collocation_points": config.num_points,
        "test_points": config.n_test_points,
        "mape": float(metrics["mape"]),
        "mae": float(metrics["mae"]),
        "mse": float(metrics["mse"]),
        "rmse": float(metrics["rmse"]),
        "max_error": float(metrics["max_error"]),
        "weights": str(config.weights),
        "T_min": phys.T_min,
        "T_max": phys.T_max,
    }
    if bc_info:
        for name, bc_type in bc_info.items():
            result[f"bc_{name}"] = bc_type

    if return_model:
        return result, history, pinn
    return result, history


def generate_test_points(geom: GeometryBase, n_points: int) -> jnp.ndarray:
    """Генерирует равномерную сетку тестовых точек."""
    dim = geom.dim
    if dim == 1:
        if hasattr(geom, "x0") and hasattr(geom, "x1"):
            return jnp.linspace(geom.x0, geom.x1, n_points).reshape(-1, 1)
        raise ValueError("Нет x0/x1 для 1D.")
    elif dim == 2:
        if hasattr(geom, "x_min") and hasattr(geom, "x_max"):
            pts = int(round(n_points ** (1 / dim)))
            x = jnp.linspace(geom.x_min, geom.x_max, pts)
            y = jnp.linspace(geom.y_min, geom.y_max, pts)
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            return jnp.column_stack([xx.ravel(), yy.ravel()])
        raise ValueError("Нет x_min/x_max для 2D.")
    elif dim == 3:
        if hasattr(geom, "x_min") and hasattr(geom, "x_max"):
            pts = int(round(n_points ** (1 / dim)))
            x = jnp.linspace(geom.x_min, geom.x_max, pts)
            y = jnp.linspace(geom.y_min, geom.y_max, pts)
            z = jnp.linspace(geom.z_min, geom.z_max, pts)
            xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")
            return jnp.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        raise ValueError("Нет x_min/x_max для 3D.")
    raise ValueError(f"Размерность {dim} не поддерживается.")


def run_grid_search(
    param_grid: Dict[str, List[Any]],
    geom: GeometryBase,
    phys: PhysicsParams,
    pde_fn: Callable,
    exact_fn: Callable,
    bc_configs: List[Dict[str, Any]],
    bc_info: Optional[Dict[str, str]] = None,
    base_config: Optional[TrainConfig] = None,
    csv_path: str = "csv_results/pinn_results.csv",
) -> pd.DataFrame:
    """Запускает Grid Search по гиперпараметрам."""
    import itertools

    if base_config is None:
        base_config = TrainConfig()

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    combinations = list(itertools.product(*values))

    results = []
    total = len(combinations)
    print(f"Grid Search: {total} комбинаций")

    for i, combo in enumerate(combinations, 1):
        print(f"Эксперимент {i}/{total}")

        config_dict = {
            "hidden_features": base_config.hidden_features,
            "num_layers": base_config.num_layers,
            "activation_name": base_config.activation_name,
            "opt_name": base_config.opt_name,
            "lr": base_config.lr,
            "epochs": base_config.epochs,
            "n_test_points": base_config.n_test_points,
            "num_points": base_config.num_points,
            "weights": base_config.weights,
            "log_interval": base_config.log_interval,
        }
        for key, value in zip(keys, combo):
            if key in config_dict:
                config_dict[key] = value
        config = TrainConfig(**config_dict)

        try:
            result, _ = run_experiment(
                config=config,
                geom=geom,
                phys=phys,
                pde_fn=pde_fn,
                exact_fn=exact_fn,
                bc_configs=bc_configs,
                bc_info=bc_info,
                return_model=False,
            )
            results.append(result)
        except Exception as e:
            print(f"Ошибка: {e}")
            continue

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=header, index=False)
        print(f"Результаты сохранены в {csv_path}")
        return df
    else:
        print("Нет результатов.")
        return pd.DataFrame()