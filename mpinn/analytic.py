import jax.numpy as jnp

def line_1d_dirichlet_exact(x, phys):
    x0, x1 = phys.x_left, phys.x_right
    T0, T1 = phys.T_left, phys.T_right
    return T0 + (T1 - T0) * (x - x0) / (x1 - x0)

def line_1d_neuman_exact(x, phys):
    return phys.T_left + phys.grad_right * (x - phys.x_left)

def line_1d_robin_exact(x, phys):
    x0 = phys.x_left
    x1 = phys.x_right

    h = phys.h
    _lambda = phys._lambda
    T_inf = phys.T_inf
    T_left = phys.T_left

    # Условие Робина: h·T + λ·dT/dx = h·T_inf
    # Общее решение: T(x) = A·x + B
    # Из левой границы: B = T_left - A·x0
    # Подстановка в правую границу даёт:
    denominator = h * (x1 - x0) + _lambda
    A = h * (T_inf - T_left) / denominator
    B = T_left - A * x0

    return A * x + B

def cylinder_1d_dirichlet_exact(x, phys):
    r0, r1 = phys.x_left, phys.x_right
    T0, T1 = phys.T_left, phys.T_right
    ln_r0 = jnp.log(r0)
    ln_r1 = jnp.log(r1)
    return T0 + (T1 - T0) * (jnp.log(x) - ln_r0) / (ln_r1 - ln_r0)

def cylinder_1d_neuman_exact(x, phys):
    r0, r1 = phys.x_left, phys.x_right
    # Константа интегрирования: для цилиндра dT/dr = C/r
    # Из условия Неймана: C = grad_right * r1
    C = phys.grad_right * r1
    return phys.T_left + C * (jnp.log(x) - jnp.log(r0))

def cylinder_1d_robin_exact(x, phys):
    r0 = phys.x_left
    r1 = phys.x_right

    h = phys.h
    _lambda = phys._lambda
    T_left = phys.T_left
    T_inf = phys.T_inf

    # Константа интегрирования, найденная из условия конвекции на r = r1
    denominator = _lambda + h * r1 * jnp.log(r1 / r0)
    C1 = h * r1 * (T_inf - T_left) / denominator

    # Общее решение: T(r) = T_left + C1 * ln(r/r0)
    return T_left + C1 * jnp.log(x / r0)

def sphere_1d_dirichlet_exact(x, phys):
    r0, r1 = phys.x_left, phys.x_right
    T0, T1 = phys.T_left, phys.T_right
    inv_r0 = 1.0 / r0
    inv_r1 = 1.0 / r1
    inv_x = 1.0 / x
    return T0 + (T1 - T0) * (inv_r0 - inv_x) / (inv_r0 - inv_r1)

def sphere_1d_neuman_exact(x, phys):
    r0 = phys.x_left
    r1 = phys.x_right
    return phys.T_left - phys.grad_right * r1**2 * (1.0 / x - 1.0 / r0)

def sphere_1d_robin_exact(x, phys):
    r0 = phys.x_left
    r1 = phys.x_right

    h = phys.h
    _lambda = phys._lambda
    T_left = phys.T_left
    T_inf = phys.T_inf

    # Общая форма решения для сферы: T(r) = T_left + C1 * (1/r0 - 1/r)
    # Производная: dT/dr = C1 / r^2
    
    # Условие Робина на r1: -_lambda * dT/dr = h * (T - T_inf)
    # -_lambda * (C1 / r1^2) = h * (T_left + C1*(1/r0 - 1/r1) - T_inf)
    # C1 * [_lambda/r1^2 + h*(1/r0 - 1/r1)] = h * (T_inf - T_left)
    
    denominator = (_lambda / r1**2) + h * (1.0/r0 - 1.0/r1)
    C1 = h * (T_inf - T_left) / denominator

    return T_left + C1 * (1.0/r0 - 1.0/x)