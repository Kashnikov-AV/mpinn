import jax
import jax.numpy as jnp

# Векторизованная версия line_1d для обработки всего массива точек сразу
@jax.jit
def line_1d(model, x, phys):
    """
    Вычисляет невязку PDE для 1D задачи (уравнение d2y/dx2 = 0).
    Использует vmap для векторизации вычисления производных по всем точкам.
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Массив координат формы (n_points, 1).
    phys : object
        Физические параметры (не используются для этого уравнения).
    
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
    d2y_dx2 = jax.vmap(hessian_predict)(x_flat)
    
    return jnp.mean(d2y_dx2 ** 2)


@jax.jit
def cylinder_1d(model, x, phys):
    """
    Вычисляет невязку PDE для цилиндрической геометрии.
    Уравнение: dT/dr * (1/r) + d2T/dr2 = 0
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Массив радиальных координат формы (n_points, 1).
    phys : object
        Физические параметры (не используются для этого уравнения).
    
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
    return jnp.mean(residual ** 2)


@jax.jit
def sphere_1d(model, x, phys):
    """
    Вычисляет невязку PDE для сферической геометрии.
    Уравнение: d2y/dr2 + (2/r) * dy/dr = 0
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Массив радиальных координат формы (n_points, 1).
    phys : object
        Физические параметры (не используются для этого уравнения).
    
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
    return jnp.mean(residual ** 2)