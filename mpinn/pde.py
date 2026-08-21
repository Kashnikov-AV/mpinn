import jax
import jax.numpy as jnp
from typing import Optional, Callable


# Векторизованная версия line_1d для обработки всего массива точек сразу
def line_1d(model, x, phys):
    """
    Вычисляет невязку PDE для 1D задачи.
    Без источника: d2T/dx2 = 0
    С источником: d2T/dx2 + f(x) = 0
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Массив координат формы (n_points, 1).
    phys : object
        Физические параметры, должен содержать source_fn.
    
    Returns:
    --------
    float
        Средняя квадратичная невязка.
    """
    # Векторизуем вычисление функции сети
    def predict(x_val):
        return model(jnp.atleast_2d(x_val)).ravel()[0]
    
    # Создаём векторизованные версии первой и второй производных
    grad_predict = jax.grad(predict)
    hessian_predict = jax.grad(grad_predict)
    
    # Применяем vmap для вычисления на всех точках одновременно
    x_flat = x.ravel()
    d2T_dx2 = jax.vmap(hessian_predict)(x_flat)
    
    # Вычисляем источник если задан
    source_val = 0.0
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_val = phys.source_fn(x_flat)
    
    # Невязка: d2T/dx2 + f(x) = 0
    residual = d2T_dx2 + source_val
    
    return jnp.mean(residual ** 2)


@jax.jit
def cylinder_1d(model, x, phys):
    """
    Вычисляет невязку PDE для цилиндрической геометрии.
    Без источника: dT/dr * (1/r) + d2T/dr2 = 0
    С источником: dT/dr * (1/r) + d2T/dr2 + f(r) = 0
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Массив радиальных координат формы (n_points, 1).
    phys : object
        Физические параметры, должен содержать source_fn.
    
    Returns:
    --------
    float
        Средняя квадратичная невязка.
    """
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]
    
    grad_predict = jax.grad(predict)
    hessian_predict = jax.grad(grad_predict)
    
    r_flat = x.ravel()
    # Защита от деления на ноль в центре цилиндра
    r_safe = jnp.where(r_flat == 0.0, 1e-8, r_flat)
    
    # Векторизованное вычисление производных
    dT_dr = jax.vmap(grad_predict)(r_flat)
    d2T_dr2 = jax.vmap(hessian_predict)(r_flat)
    
    residual = dT_dr / r_safe + d2T_dr2
    
    # Вычисляем источник если задан
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_val = phys.source_fn(r_flat)
        residual = residual + source_val
    
    return jnp.mean(residual ** 2)


@jax.jit
def sphere_1d(model, x, phys):
    """
    Вычисляет невязку PDE для сферической геометрии.
    Без источника: d2y/dr2 + (2/r) * dy/dr = 0
    С источником: d2y/dr2 + (2/r) * dy/dr + f(r) = 0
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Массив радиальных координат формы (n_points, 1).
    phys : object
        Физические параметры, должен содержать source_fn.
    
    Returns:
    --------
    float
        Средняя квадратичная невязка.
    """
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]
    
    grad_predict = jax.grad(predict)
    hessian_predict = jax.grad(grad_predict)
    
    r_flat = x.ravel()
    r_safe = jnp.where(r_flat == 0.0, 1e-8, r_flat)
    
    # Векторизованное вычисление производных
    dy_dr = jax.vmap(grad_predict)(r_flat)
    d2y_dr2 = jax.vmap(hessian_predict)(r_flat)
    
    residual = d2y_dr2 + (2.0 / r_safe) * dy_dr
    
    # Вычисляем источник если задан
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_val = phys.source_fn(r_flat)
        residual = residual + source_val
    
    return jnp.mean(residual ** 2)