#!/usr/bin/env python3
"""
Простой пример 1D задачи теплопроводности:
- domain: x ∈ [0, 1]
- Граничные условия: T(0) = 300, T(1) = 500 (Dirichlet-Dirichlet)
"""

import jax.numpy as jnp
from mpinn.config import PhysicsParams, TrainConfig
from mpinn.runner import run_experiment
from mpinn.pde import line_1d
from mpinn.analytic import line_1d_dirichlet_exact
from mpinn.bc import dirichlet_bc
from functools import partial

def main():
    # Физические параметры
    phys = PhysicsParams(
        x0=0.0,
        x1=1.0,
        T0=300.0,   # T0 = 300
        T1=500.0,  # T1 = 500
        _lambda=1.0,
        h=10.0,
        T_inf=500.0
    )
    
    # Конфигурация обучения (default) с уменьшенным числом эпох для быстрого теста
    config = TrainConfig(max_epochs=500)
    
    # Граничные условия Dirichlet для обеих границ
    bc_fns = [
        partial(dirichlet_bc, x=jnp.array([[phys.x0]]), T=phys.T0),
        partial(dirichlet_bc, x=jnp.array([[phys.x1]]), T=phys.T1),
    ]
    
    print("Запуск эксперимента...")
    print(f"Домен: x ∈ [{phys.x0}, {phys.x1}]")
    print(f"T(x0={phys.x0}) = {phys.T0} K")
    print(f"T(x1={phys.x1}) = {phys.T1} K")
    print(f"Конфигурация: {config.num_points} точек коллокации, {config.max_epochs} эпох")
    
    # Запуск эксперимента
    metrics, history = run_experiment(
        config=config,
        phys=phys,
        pde_fn=line_1d,
        exact_fn=line_1d_dirichlet_exact,
        bc_fns_override=bc_fns
    )
    
    print("\n=== Результаты ===")
    print(f"MSE: {metrics['mse']:.6e}")
    print(f"MAE: {metrics['mae']:.6e}")
    print(f"RMSE: {metrics['rmse']:.6e}")
    print(f"Max Error: {metrics['max_error']:.6e}")
    print(f"MAPE: {metrics['mape']:.6e}")
    print(f"Время обучения: {metrics['training_time']} c")
    print(f"Эпох обучено: {metrics['epochs_trained']}")

if __name__ == "__main__":
    main()
