"""
Пример использования модуля experiments для 1D задачи теплопроводности.
Демонстрирует запуск одного эксперимента и Grid Search.
"""
from experiments import (
    PhysicsParams, 
    TrainConfig, 
    run_experiment, 
    run_grid_search,
    show_plot,
    show_history,
)


def single_experiment_example():
    """Пример запуска одного эксперимента с Early Stopping."""
    
    # Физические параметры
    phys = PhysicsParams(
        x_left=0.0,
        x_right=1.0,
        T_left=300.0,
        T_inf=500.0,
        _lambda=1.0,
        h=10.0,
    )
    
    # Конфигурация обучения
    config = TrainConfig(
        hidden_features=64,
        num_layers=2,
        activation_name='GELU',
        opt_name='adam',
        lr=0.01,
        max_epochs=3000,
        patience=200,
        min_delta=1e-6,
        num_points=100,
        weights=(1.0, 1.0, 1.0),
        save_img=True,
        show_plot=True,
        image_path='images/pinn/1D_line_Robin_single.png',
    )
    
    # Запуск эксперимента
    metrics, history = run_experiment(config=config, phys=phys)
    
    print(f"\nРезультаты эксперимента:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAPE: {metrics['mape']:.6f}")
    print(f"Обучено эпох: {metrics['epochs_trained']}")
    
    # Построение графика истории обучения
    show_history(
        history=history,
        save_path='images/history_line_robin_single.png',
        show_plot=True
    )
    
    return metrics, history


def grid_search_example():
    """Пример Grid Search для перебора гиперпараметров."""
    
    phys = PhysicsParams()
    
    base_config = TrainConfig(
        hidden_features=64,
        num_layers=2,
        activation_name='GELU',
        opt_name='adam',
        lr=0.01,
        max_epochs=1000,
        patience=150,
        num_points=50,
        weights=(1.0, 1.0, 1.0),
    )
    
    # Сетка параметров для перебора
    param_grid = {
        'hidden_features': [32, 64],
        'num_layers': [1, 2],
        'activation_name': ['tanh', 'GELU'],
        'lr': [0.005, 0.01],
    }
    
    # Запуск Grid Search
    df_results = run_grid_search(
        param_grid=param_grid,
        base_config=base_config,
        phys=phys,
        csv_path='csv_results/1D_grid_search_results.csv',
    )
    
    print(f"\nРезультаты Grid Search ({len(df_results)} экспериментов):")
    print(df_results.sort_values('mse').head())
    
    return df_results


if __name__ == '__main__':
    # Пример 1: Одиночный эксперимент
    print("=" * 50)
    print("Запуск одиночного эксперимента")
    print("=" * 50)
    metrics, history = single_experiment_example()
    
    # Пример 2: Grid Search (закомментирован для быстрого старта)
    # print("\n" + "=" * 50)
    # print("Запуск Grid Search")
    # print("=" * 50)
    # df = grid_search_example()
