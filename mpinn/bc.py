"""Boundary condition functions for 1D heat conduction problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def dirichlet_bc(model, x, T):
    """
    Вычисляет среднюю квадратичную невязку граничного условия Дирихле.

    Parameters
    ----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Координаты граничных точек формы (n_points, 1).
    T : float | jax.Array
        Заданное значение температуры на границе.

    Returns
    -------
    float
        Средняя квадратичная невязка (1/N * sum((T_pred - T)^2)).
    """

    # Векторизованное предсказание сети
    def predict(xi):
        return model(jnp.atleast_2d(xi)).ravel()[0]

    # Вычисляем невязки для всех точек сразу через vmap
    residuals = jax.vmap(lambda xi: (predict(xi) - T) ** 2)(x.ravel())
    return jnp.mean(residuals)


@jax.jit
def neuman_bc(model, x, g):
    """
    Вычисляет среднюю квадратичную невязку граничного условия Неймана.

    Parameters
    ----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Координаты граничных точек формы (n_points, 1).
    g : float | jax.Array
        Заданное значение градиента температуры на границе.

    Returns
    -------
    float
        Средняя квадратичная невязка (1/N * sum((dT/dx - g)^2)).
    """

    def predict(xi):
        return model(jnp.atleast_2d(xi)).ravel()[0]

    # Векторизованный градиент
    grad_predict = jax.grad(predict)

    # Вычисляем невязки для всех точек
    residuals = jax.vmap(lambda xi: (grad_predict(xi) - g) ** 2)(x.ravel())
    return jnp.mean(residuals)


@jax.jit
def robin_bc(model, x, alpha, beta, h):
    """
    Вычисляет среднюю квадратичную невязку граничного условия Робина.

    Parameters
    ----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Координаты граничных точек формы (n_points, 1).
    alpha : float
        Коэффициент при температуре.
    beta : float
        Коэффициент при градиенте температуры.
    h : float
        Заданное значение комбинации.

    Returns
    -------
    float
        Средняя квадратичная невязка (1/N * sum((alpha*T + beta*dT/dx - h)^2)).
    """

    def predict(xi):
        return model(jnp.atleast_2d(xi)).ravel()[0]

    grad_predict = jax.grad(predict)

    # Векторизованное вычисление невязки
    def compute_residual(xi):
        T_val = predict(xi)
        dT_dx = grad_predict(xi)
        return (alpha * T_val + beta * dT_dx - h) ** 2

    residuals = jax.vmap(compute_residual)(x.ravel())
    return jnp.mean(residuals)
