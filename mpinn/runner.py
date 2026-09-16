"""
Запуск экспериментов PINN с Grid Search.
Поддерживает 1D, 2D, 3D.
"""

import os
import time
import math
import gc
import pandas as pd

from typing import Any, Callable, Dict, List, Optional
import jax
import jax.numpy as jnp
from flax import nnx

from mpinn.config import PhysicsParams, TrainConfig, get_activation, get_optimizer
from mpinn.config import normalize_coords, denormalize_temp
from mpinn.geom import GeometryBase
from mpinn.pinn_core import PINN, FCNet


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


class ScaledNet(nnx.Module):
    """Обёртка: нормализует вход, денормализует выход (T = T_norm * T_max)."""

    def __init__(self, base, x_min, x_max, T_max):
        self.base = base
        self.x_min = x_min
        self.x_max = x_max
        self.T_max = T_max

    def __call__(self, x):
        x_norm = normalize_coords(x, self.x_min, self.x_max)
        return denormalize_temp(self.base(x_norm), self.T_max)


def run_experiment(
    config: TrainConfig,
    geom: GeometryBase,
    phys: PhysicsParams,
    pde_fn: Callable,
    exact_fn: Callable,
    bc_configs: List[Dict],
    bc_info: Optional[Dict] = None,
    geometry_type: Optional[str] = None,
    x_test: Optional[jnp.ndarray] = None,
    target_rl2: float = 1e-3,
    block_epochs: int = 50,
    max_epochs: int = 5000,
    log_interval: int = 10,
    return_model: bool = False,
):
    """
    Адаптивное обучение PINN блоками до достижения целевого RL2.

    Обучение идёт блоками по block_epochs эпох. После каждого блока
    вычисляется RL2 на тестовых точках. При достижении target_rl2
    обучение останавливается, и возвращается лучшая модель.
    """
    # ---------- 1. Границы области ----------
    if hasattr(geom, "bounds"):
        x_min, x_max = geom.bounds
        if geom.dim == 1:
            x_min, x_max = x_min[0], x_max[0]
    elif hasattr(geom, "x0") and hasattr(geom, "x1"):
        x_min, x_max = geom.x0, geom.x1
    else:
        raise ValueError("Не удалось определить границы области.")
    T_max = phys.T_max

    # ---------- 2. Точки коллокации и теста ----------
    x_colloc = geom.sample_interior(config.num_points, rng=jax.random.PRNGKey(0))
    if x_test is None:
        x_test = generate_test_points(geom, config.n_test_points)

    # ---------- 3. Модель ----------
    base = FCNet(
        geom.dim, config.hidden_features, 1, config.num_layers,
        get_activation(config.activation_name), nnx.Rngs(0),
    )
    model = ScaledNet(base, x_min, x_max, T_max)

    pinn = PINN(
        model,
        get_optimizer(config.opt_name, config.lr),
        tuple(config.weights),
        phys,
        pde_fn,
        bc_configs,
    )

    # ---------- 4. Функция оценки ----------
    @jax.jit
    def evaluate(params):
        model = nnx.merge(pinn.graphdef, params)
        T_pred = model(x_test).ravel()
        T_exact = exact_fn(x_test, phys).ravel()
        return compute_rl2(T_pred, T_exact)

    # ---------- 5. Адаптивное обучение блоками ----------
    history: Dict[str, list] = {}
    total_epochs = 0
    best_rl2 = math.inf
    best_epoch = 0
    best_params = pinn.params

    start_time = time.perf_counter()
    block = block_epochs

    while total_epochs < max_epochs:
        hist, _ = pinn.fit(x_colloc, block, log_interval)
        total_epochs += block
        
        for key, values in hist.items():
            history.setdefault(key, []).extend(values)

        rl2 = evaluate(pinn.params)
        
        if rl2 < best_rl2:
            best_rl2 = rl2
            best_epoch = total_epochs
            best_params = pinn.params

        if rl2 < target_rl2:
            print(f"Целевая точность достигнута на эпохе {total_epochs}")
            break

    total_time = time.perf_counter() - start_time

    # ---------- 6. Восстановление лучшей модели ----------
    pinn.params = best_params

    # ---------- 7. Финальные метрики ----------
    final_rl2 = float(evaluate(pinn.params))
    T_pred = pinn.predict(x_test).ravel()
    T_exact = exact_fn(x_test, phys).ravel()

    mae = float(jnp.mean(jnp.abs(T_pred - T_exact)))
    mse = float(jnp.mean((T_pred - T_exact) ** 2))
    rmse = float(jnp.sqrt(mse))
    max_err = float(jnp.max(jnp.abs(T_pred - T_exact)))
    mape = float(jnp.mean(jnp.abs((T_pred - T_exact) / (jnp.abs(T_exact) + 1e-8))))

    # ---------- 8. Формирование результата ----------
    result = {
        "total_epochs": total_epochs,           # сколько всего прошло эпох
        "best_epoch": best_epoch,               # эпоха с минимальным RL2
        "final_rl2": float(final_rl2),          # RL2 после восстановления модели
        "training_time_sec": total_time,        # общее время обучения, сек
        "lr": config.lr,
        "activation_func": config.activation_name,
        "layers": config.num_layers,
        "neurons": config.hidden_features,
        "optimizer": config.opt_name,
        "collocation_points": config.num_points,
        "test_points": config.n_test_points,
        "mape": mape,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_err,
        "weights": str(config.weights),
    }
    if bc_info:
        for name, bc_type in bc_info.items():
            result[f"bc_{name}"] = bc_type
    if geometry_type:
        result["geometry_type"] = geometry_type

    print(f"Лучший RL2: {best_rl2:.2e} на эпохе {best_epoch} | Всего эпох: {total_epochs}")

    if return_model:
        return result,  history, pinn
    return result, history


def compute_rl2(T_pred, T_exact, eps: float = 1e-12) -> float:
    """Относительная L2-ошибка."""
    T_pred = T_pred.ravel()
    T_exact = T_exact.ravel()
    return jnp.linalg.norm(T_pred - T_exact) / (jnp.linalg.norm(T_exact) + eps)


def run_grid_search(
    param_grid: Dict[str, List[Any]],
    geom: GeometryBase,
    phys: PhysicsParams,
    pde_fn: Callable,
    exact_fn: Callable,
    bc_configs: List[Dict[str, Any]],
    bc_info: Optional[Dict[str, str]] = None,
    geometry_type: Optional[str] = None,
    base_config: Optional[TrainConfig] = None,
    csv_path: str = "csv_results/pinn_results.csv",
    target_rl2: float = 1e-3,
    block_epochs: int = 50,
    max_epochs: int = 5000,
    log_interval: int = 100,
) -> pd.DataFrame:
    """Адаптивный Grid Search по гиперпараметрам."""
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
            "epochs": base_config.epochs,        # не используется в адаптивном режиме
            "n_test_points": base_config.n_test_points,
            "num_points": base_config.num_points,
            "weights": base_config.weights,
            "log_interval": log_interval,
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
                geometry_type=geometry_type,
                target_rl2=target_rl2,
                block_epochs=block_epochs,
                max_epochs=max_epochs,
                log_interval=log_interval,
                return_model=False,
            )
            results.append(result)
        except Exception as e:
            print(f"Ошибка: {e}")
            continue
        finally:
            gc.collect()

        if i % 10 == 0:
            jax.clear_caches()

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