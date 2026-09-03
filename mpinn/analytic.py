"""
Точные аналитические решения для 1D задач теплопроводности.
Каждая функция возвращает вызываемый объект exact(x, phys),
где phys используется для получения _lambda (если нужно).
Параметры задаются явно при создании фабрики.
"""

import jax.numpy as jnp


def line_1d_dirichlet_exact(T0: float, T1: float, x0: float, x1: float):
    """T(x) = T0 + (T1 - T0) * (x - x0) / (x1 - x0)"""
    def exact(x, phys):
        return T0 + (T1 - T0) * (x - x0) / (x1 - x0)
    return exact


def line_1d_neuman_exact(T0: float, flux: float, x0: float, x1: float):
    """
    Условие Неймана на правой границе: -λ * dT/dx = flux.
    Решение: T(x) = T0 + grad * (x - x0), где grad = -flux / λ.
    """
    def exact(x, phys):
        lam = phys._lambda
        grad = -flux / lam
        return T0 + grad * (x - x0)
    return exact


def line_1d_robin_exact(T0: float, T_inf: float, h: float, x0: float, x1: float):
    """
    Условие Робина на правой границе: h*(T - T_inf) + λ*dT/dx = 0.
    Решение: T(x) = T0 + A*(x - x0), A = h*(T_inf - T0) / (h*(x1 - x0) + λ)
    """
    def exact(x, phys):
        lam = phys._lambda
        denominator = h * (x1 - x0) + lam
        A = h * (T_inf - T0) / denominator
        return T0 + A * (x - x0)
    return exact


def cylinder_1d_dirichlet_exact(T0: float, T1: float, r0: float, r1: float):
    """T(r) = T0 + (T1 - T0) * ln(r/r0) / ln(r1/r0)"""
    def exact(x, phys):
        return T0 + (T1 - T0) * jnp.log(x / r0) / jnp.log(r1 / r0)
    return exact


def cylinder_1d_neuman_exact(T0: float, flux: float, r0: float, r1: float):
    """
    Условие Неймана на правой границе: -λ * dT/dr = flux.
    Решение: T(r) = T0 + grad * r1 * ln(r/r0), где grad = -flux / λ.
    """
    def exact(x, phys):
        lam = phys._lambda
        grad = -flux / lam
        C = grad * r1
        return T0 + C * jnp.log(x / r0)
    return exact


def cylinder_1d_robin_exact(T0: float, T_inf: float, h: float, r0: float, r1: float):
    """
    Условие Робина на правой границе: h*(T - T_inf) + λ*dT/dr = 0.
    Решение: T(r) = T0 + C1 * ln(r/r0),
              C1 = h*r1*(T_inf - T0) / (λ + h*r1*ln(r1/r0))
    """
    def exact(x, phys):
        lam = phys._lambda
        denominator = lam + h * r1 * jnp.log(r1 / r0)
        C1 = h * r1 * (T_inf - T0) / denominator
        return T0 + C1 * jnp.log(x / r0)
    return exact


def sphere_1d_dirichlet_exact(T0: float, T1: float, r0: float, r1: float):
    """T(r) = T0 + (T1 - T0) * (1/r0 - 1/r) / (1/r0 - 1/r1)"""
    def exact(x, phys):
        inv_r0 = 1.0 / r0
        inv_r1 = 1.0 / r1
        inv_x = 1.0 / x
        return T0 + (T1 - T0) * (inv_r0 - inv_x) / (inv_r0 - inv_r1)
    return exact


def sphere_1d_neuman_exact(T0: float, flux: float, r0: float, r1: float):
    """
    Условие Неймана на правой границе: -λ * dT/dr = flux.
    Решение: T(r) = T0 - grad * r1^2 * (1/r - 1/r0),
              где grad = -flux / λ.
    """
    def exact(x, phys):
        lam = phys._lambda
        grad = -flux / lam
        return T0 - grad * r1**2 * (1.0 / x - 1.0 / r0)
    return exact


def sphere_1d_robin_exact(T0: float, T_inf: float, h: float, r0: float, r1: float):
    """
    Условие Робина на правой границе: h*(T - T_inf) + λ*dT/dr = 0.
    Решение: T(r) = T0 + C1 * (1/r0 - 1/r),
              C1 = h*(T_inf - T0) / (λ/r1^2 + h*(1/r0 - 1/r1))
    """
    def exact(x, phys):
        lam = phys._lambda
        denominator = (lam / r1**2) + h * (1.0 / r0 - 1.0 / r1)
        C1 = h * (T_inf - T0) / denominator
        return T0 + C1 * (1.0 / r0 - 1.0 / x)
    return exact