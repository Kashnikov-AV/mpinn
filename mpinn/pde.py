"""
Функции невязки PDE для задач теплопроводности.
Поддерживают как постоянную, так и переменную теплопроводность (функцию от температуры).
Все функции возвращают скаляр (средний квадрат невязки).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ==================== 1D задачи ====================

@jax.jit
def line_1d(model, x, phys):
    """
    1D уравнение теплопроводности в декартовых координатах.
    λ(T) * d²T/dx² + λ'(T) * (dT/dx)² + f(x) = 0
    """
    def predict(x_val):
        return model(jnp.atleast_2d(x_val)).ravel()[0]

    grad_T = jax.grad(predict)
    def grad_T_fn(x_val):
        return grad_T(x_val)
    d2T_dx2 = jax.vmap(jax.grad(grad_T_fn))(x.ravel())
    dT_dx = jax.vmap(grad_T)(x.ravel())

    T_pred = model(x).ravel()

    # Определяем lambda и её производную
    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    laplacian = d2T_dx2
    grad_sq = dT_dx ** 2

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(x.ravel())

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)


@jax.jit
def cylinder_1d(model, x, phys):
    """
    1D уравнение теплопроводности в цилиндрических координатах (осесимметричное).
    λ(T) * (d²T/dr² + (1/r) dT/dr) + λ'(T) * (dT/dr)² + f(r) = 0
    """
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]

    grad_T = jax.grad(predict)
    hess_T = jax.grad(grad_T)

    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()

    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    laplacian = d2T_dr2 + (1.0 / r_safe) * dT_dr
    grad_sq = dT_dr ** 2

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(r)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)


@jax.jit
def sphere_1d(model, x, phys):
    """
    1D уравнение теплопроводности в сферических координатах (сферически-симметричное).
    λ(T) * (d²T/dr² + (2/r) dT/dr) + λ'(T) * (dT/dr)² + f(r) = 0
    """
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]

    grad_T = jax.grad(predict)
    hess_T = jax.grad(grad_T)

    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()

    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    laplacian = d2T_dr2 + (2.0 / r_safe) * dT_dr
    grad_sq = dT_dr ** 2

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(r)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)


# ==================== 2D задачи ====================

@jax.jit
def laplace_2d(model, x, phys):
    """
    2D уравнение теплопроводности в декартовых координатах.
    λ(T) * (∂²T/∂x² + ∂²T/∂y²) + λ'(T) * ((∂T/∂x)² + (∂T/∂y)²) + f(x,y) = 0
    """
    def predict(x_pts):
        return model(jnp.atleast_2d(x_pts)).ravel()[0]

    grad_fn = jax.grad(predict)

    def laplacian_pt(x_pt):
        g = grad_fn(x_pt)
        d2x = jax.grad(lambda p: grad_fn(p)[0])(x_pt)
        d2y = jax.grad(lambda p: grad_fn(p)[1])(x_pt)
        return d2x + d2y

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)  # (N,2)

    T_pred = model(x).ravel()

    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(x)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)


@jax.jit
def polar_2d(model, x, phys):
    """
    2D уравнение теплопроводности в полярных координатах (осесимметричное).
    λ(T) * (d²T/dr² + (1/r) dT/dr) + λ'(T) * (dT/dr)² + f(r) = 0
    Модель принимает на вход точки (r, θ), но θ игнорируется.
    """
    r = x[:, 0]

    def predict_radial(r_val):
        # формируем точку (r, 0) для модели
        pt = jnp.array([[r_val, 0.0]])
        return model(pt).ravel()[0]

    grad_T = jax.grad(predict_radial)
    hess_T = jax.grad(grad_T)

    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    # Для T_pred используем модель на полных точках (r, θ)
    T_pred = model(x).ravel()

    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    laplacian = d2T_dr2 + (1.0 / r_safe) * dT_dr
    grad_sq = dT_dr ** 2

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(x)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)


# ==================== 3D задачи ====================

@jax.jit
def laplace_3d(model, x, phys):
    """
    3D уравнение теплопроводности в декартовых координатах.
    λ(T) * (∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²) + λ'(T) * ((∂T/∂x)² + (∂T/∂y)² + (∂T/∂z)²) + f(x,y,z) = 0
    """
    def predict(x_pts):
        return model(jnp.atleast_2d(x_pts)).ravel()[0]

    grad_fn = jax.grad(predict)

    def laplacian_pt(x_pt):
        g = grad_fn(x_pt)
        d2x = jax.grad(lambda p: grad_fn(p)[0])(x_pt)
        d2y = jax.grad(lambda p: grad_fn(p)[1])(x_pt)
        d2z = jax.grad(lambda p: grad_fn(p)[2])(x_pt)
        return d2x + d2y + d2z

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)  # (N,3)

    T_pred = model(x).ravel()

    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(x)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)


@jax.jit
def cylinder_3d_axisymmetric(model, x, phys):
    """
    3D уравнение теплопроводности в цилиндрических координатах (осесимметричное).
    λ(T) * (∂²T/∂r² + (1/r)∂T/∂r + ∂²T/∂z²) + λ'(T) * ((∂T/∂r)² + (∂T/∂z)²) + f(r,z) = 0
    Модель принимает на вход точки (r, z).
    """
    def predict(rz):
        return model(rz).ravel()[0]

    grad_fn = jax.grad(predict)

    def laplacian_pt(rz_pt):
        g = grad_fn(rz_pt)  # [∂T/∂r, ∂T/∂z]
        d2r = jax.grad(lambda p: grad_fn(p)[0])(rz_pt)
        d2z = jax.grad(lambda p: grad_fn(p)[1])(rz_pt)
        r = rz_pt[0]
        r_safe = jnp.where(r == 0.0, 1e-12, r)
        return d2r + (1.0 / r_safe) * g[0] + d2z

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)  # (N,2)

    T_pred = model(x).ravel()

    if callable(phys._lambda):
        lambda_vals = jax.vmap(phys._lambda)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(phys._lambda))(T_pred)
    else:
        lambda_vals = phys._lambda * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)

    source_val = 0.0
    if phys.source_fn is not None:
        source_val = phys.source_fn(x)

    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + source_val
    return jnp.mean(residual ** 2)