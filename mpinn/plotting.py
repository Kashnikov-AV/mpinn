"""
Функции отрисовки графиков в стиле pinn_core.
"""
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Dict, List, Optional
import os


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


def show_history(history: Dict[str, List[float]], save_path: Optional[str] = None, show_plot: bool = True):
    """
    Отображает и/или сохраняет график истории обучения (потери по эпохам).
    Стиль соответствует оригинальному скрипту пользователя.
    
    Параметры:
    - history: словарь с ключами ['steps', 'pde', 'bc_0', 'bc_1', 'total_loss']
    - save_path: путь для сохранения файла (если None, не сохраняется)
    - show_plot: если True, отображает график
    """
    plt.figure(figsize=(12, 7))
    steps = history['steps']
    
    # PDE потери
    plt.semilogy(steps, history['pde'], 'b', label='PDE Потери', linewidth=2)
    # Граничные условия
    plt.semilogy(steps, history['bc_0'], 'r--', label='ГУ слева', linewidth=2, alpha=0.7)
    plt.semilogy(steps, history['bc_1'], 'r--', label='ГУ справа', linewidth=2, alpha=0.7)
    # Общая ошибка
    plt.semilogy(steps, history['total_loss'], 'g--', label='Общая ошибка', linewidth=2, alpha=0.7)
    
    plt.title('История Обучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        plt.savefig(save_path, dpi=72, bbox_inches='tight')
        print(f"График истории сохранен: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
