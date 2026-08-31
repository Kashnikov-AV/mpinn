"""Boundary condition functions for 1D/2D/3D problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def dirichlet_bc(model, points, values):
    """
    Вычисляет невязки граничного условия Дирихле.

    Parameters
    ----------
    model : nnx.Module
        Нейронная сеть.
    points : jax.Array
        Координаты граничных точек формы (n_points, D).
    values : float | jax.Array
        Заданное значение температуры на границе.

    Returns
    -------
    jax.Array
        Массив невязок формы (n_points,).
    """
    T_pred = model(points).ravel()
    return T_pred - values


@jax.jit
def neumann_bc(model, points, normals, flux_values):
    """
    Вычисляет невязки граничного условия Неймана.

    Parameters
    ----------
    model : nnx.Module
        Нейронная сеть.
    points : jax.Array
        Координаты граничных точек формы (n_points, D).
    normals : jax.Array
        Единичные нормали формы (n_points, D).
    flux_values : float | jax.Array
        Заданное значение потока на границе.

    Returns
    -------
    jax.Array
        Массив невязок формы (n_points,).
    """
    def predict(x):
        return model(jnp.atleast_2d(x)).ravel()[0]

    def grad_predict(x):
        return jax.grad(predict)(x)

    grads = jax.vmap(grad_predict)(points)
    normal_deriv = jnp.sum(grads * normals, axis=1)
    return normal_deriv - flux_values


@jax.jit
def robin_bc(model, points, normals, alpha, beta, value):
    """
    Вычисляет невязки граничного условия Робина.

    Parameters
    ----------
    model : nnx.Module
        Нейронная сеть.
    points : jax.Array
        Координаты граничных точек формы (n_points, D).
    normals : jax.Array
        Единичные нормали формы (n_points, D).
    alpha : float
        Коэффициент при температуре.
    beta : float
        Коэффициент при градиенте температуры.
    value : float
        Заданное значение комбинации.

    Returns
    -------
    jax.Array
        Массив невязок формы (n_points,).
    """
    def predict(x):
        return model(jnp.atleast_2d(x)).ravel()[0]

    def grad_predict(x):
        return jax.grad(predict)(x)

    grads = jax.vmap(grad_predict)(points)
    T_vals = jax.vmap(predict)(points)
    normal_deriv = jnp.sum(grads * normals, axis=1)
    return alpha * T_vals + beta * normal_deriv - value
