"""
Функции отрисовки графиков в стиле pinn_core.
"""
import matplotlib.pyplot as plt
import jax.numpy as jnp


def show_plot(x_test, T_pred, T_exact, phys, title="Сравнение PINN и точного решения"):
    """
    Отображает график сравнения предсказания сети и точного решения.
    Стиль полностью соответствует pinn_core.show_plot.
    
    Параметры:
    - x_test: тестовые координаты
    - T_pred: предсказанные значения температуры
    - T_exact: точные значения температуры
    - phys: объект PhysicsParams (для границ)
    - title: заголовок графика
    """
    plt.figure(figsize=(8, 5))
    
    # Построение графиков в стиле pinn_core
    plt.plot(x_test, T_exact, 'b-', label='Аналитическое решение', linewidth=2)
    plt.plot(x_test, T_pred, 'r:', label='ФИНС', linewidth=6)
    
    # Оформление
    plt.title(title)
    plt.xlabel('x, м')
    plt.ylabel('T, К')
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Установка границ
    plt.xlim(phys.x_left, phys.x_right)
    
    plt.tight_layout()
    plt.show()


def save_plot(x_test, T_pred, T_exact, phys, save_path, title="Сравнение PINN и точного решения"):
    """
    Сохраняет график сравнения предсказания сети и точного решения.
    Стиль полностью соответствует pinn_core.save_plot.
    
    Параметры:
    - x_test: тестовые координаты
    - T_pred: предсказанные значения температуры
    - T_exact: точные значения температуры
    - phys: объект PhysicsParams (для границ)
    - save_path: путь для сохранения файла
    - title: заголовок графика
    """
    plt.figure(figsize=(8, 5))
    
    # Построение графиков в стиле pinn_core
    plt.plot(x_test, T_exact, 'b-', label='Аналитическое решение', linewidth=2)
    plt.plot(x_test, T_pred, 'r:', label='ФИНС', linewidth=6)
    
    # Оформление
    plt.title(title)
    plt.xlabel('x, м')
    plt.ylabel('T, К')
    plt.legend(loc='best', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Установка границ
    plt.xlim(phys.x_left, phys.x_right)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=72, bbox_inches='tight')
    plt.close()
