"""
Точные аналитические решения для 1D задач теплопроводности.
Каждая фабрика возвращает вызываемый объект exact(x).
"""

import jax.numpy as jnp


def line_1d_dirichlet_exact(T0: float, T1: float, x0: float, x1: float):
    def exact(x):
        return T0 + (T1 - T0) * (x - x0) / (x1 - x0)
    return exact


def line_1d_neuman_exact(T0: float, flux: float, x0: float, x1: float, lam: float):
    grad = -flux / lam

    def exact(x):
        return T0 + grad * (x - x0)
    return exact


def line_1d_robin_exact(T0: float, T_inf: float, h: float, x0: float, x1: float, lam: float):
    A = h * (T_inf - T0) / (h * (x1 - x0) + lam)

    def exact(x):
        return T0 + A * (x - x0)
    return exact


def cylinder_1d_dirichlet_exact(T0: float, T1: float, r0: float, r1: float):
    def exact(x):
        return T0 + (T1 - T0) * jnp.log(x / r0) / jnp.log(r1 / r0)
    return exact


def cylinder_1d_neuman_exact(T0: float, flux: float, r0: float, r1: float, lam: float):
    C = (-flux / lam) * r1

    def exact(x):
        return T0 + C * jnp.log(x / r0)
    return exact


def cylinder_1d_robin_exact(T0: float, T_inf: float, h: float, r0: float, r1: float, lam: float):
    C1 = h * r1 * (T_inf - T0) / (lam + h * r1 * jnp.log(r1 / r0))

    def exact(x):
        return T0 + C1 * jnp.log(x / r0)
    return exact


def sphere_1d_dirichlet_exact(T0: float, T1: float, r0: float, r1: float):
    inv_r0 = 1.0 / r0
    inv_r1 = 1.0 / r1

    def exact(x):
        return T0 + (T1 - T0) * (inv_r0 - 1.0 / x) / (inv_r0 - inv_r1)
    return exact


def sphere_1d_neuman_exact(T0: float, flux: float, r0: float, r1: float, lam: float):
    grad = -flux / lam

    def exact(x):
        return T0 - grad * r1**2 * (1.0 / x - 1.0 / r0)
    return exact


def sphere_1d_robin_exact(T0: float, T_inf: float, h: float, r0: float, r1: float, lam: float):
    C1 = h * (T_inf - T0) / ((lam / r1**2) + h * (1.0 / r0 - 1.0 / r1))

    def exact(x):
        return T0 + C1 * (1.0 / r0 - 1.0 / x)
    return exact