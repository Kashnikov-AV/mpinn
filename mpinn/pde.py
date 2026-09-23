"""
Функции невязки PDE для задач теплопроводности.
Все функции возвращают массив невязок (N,).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _lambda_and_derivative(T_pred, lam):
    """Возвращает (λ(T), dλ/dT). lam — скаляр или callable(T)."""
    if callable(lam):
        lambda_vals = jax.vmap(lam)(T_pred)
        dlambda_dT = jax.vmap(jax.grad(lam))(T_pred)
    else:
        lambda_vals = lam * jnp.ones_like(T_pred)
        dlambda_dT = jnp.zeros_like(T_pred)
    return lambda_vals, dlambda_dT


def _source(x, source_fn):
    """source_fn — скаляр (обычно 0.0) или callable(x)."""
    if callable(source_fn):
        return source_fn(x)
    return source_fn  # скаляр, обычно 0.0


# ==================== 1D задачи ====================

def line_1d(model, x, lam, source_fn=0.0):
    """λ(T)*T'' + λ'(T)*(T')² + f(x) = 0"""
    def predict(x_val):
        return model(jnp.atleast_2d(x_val)).ravel()[0]

    grad_T = jax.grad(predict)
    dT_dx = jax.vmap(grad_T)(x.ravel())
    d2T_dx2 = jax.vmap(jax.grad(grad_T))(x.ravel())

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    residual = lambda_vals * d2T_dx2 + dlambda_dT * dT_dx**2 + _source(x.ravel(), source_fn)
    return residual


def cylinder_1d(model, x, lam, source_fn=0.0):
    """λ(T)*(T'' + T'/r) + λ'(T)*(T')² + f(r) = 0"""
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]

    grad_T = jax.grad(predict)
    hess_T = jax.grad(grad_T)

    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    laplacian = d2T_dr2 + (1.0 / r_safe) * dT_dr
    residual = lambda_vals * laplacian + dlambda_dT * dT_dr**2 + _source(r, source_fn)
    return residual


def sphere_1d(model, x, lam, source_fn=0.0):
    """λ(T)*(T'' + 2T'/r) + λ'(T)*(T')² + f(r) = 0"""
    def predict(r_val):
        return model(jnp.atleast_2d(r_val)).ravel()[0]

    grad_T = jax.grad(predict)
    hess_T = jax.grad(grad_T)

    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    laplacian = d2T_dr2 + (2.0 / r_safe) * dT_dr
    residual = lambda_vals * laplacian + dlambda_dT * dT_dr**2 + _source(r, source_fn)
    return residual


# ==================== 2D задачи ====================
def laplace_2d(model, x, lam, source_fn=0.0):
    """λ(T)*(T_xx + T_yy) + λ'(T)*|∇T|² + f(x,y) = 0"""
    def predict(x_pts):
        return model(jnp.atleast_2d(x_pts)).ravel()[0]

    grad_fn = jax.grad(predict)

    def d2dx(x_pt):
        return jax.grad(lambda p: grad_fn(p)[0])(x_pt)

    def d2dy(x_pt):
        return jax.grad(lambda p: grad_fn(p)[1])(x_pt)

    laplacian = jax.vmap(d2dx)(x) + jax.vmap(d2dy)(x)
    grad_vals = jax.vmap(grad_fn)(x)                    # (N, 2)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    grad_sq = jnp.sum(grad_vals ** 2, axis=1)           # (N,)
    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual


def polar_2d(model, x, lam, source_fn=0.0):
    """λ(T)*(T'' + T'/r) + λ'(T)*(T')² + f(r) = 0"""
    r = x[:, 0]

    def predict_radial(r_val):
        pt = jnp.array([[r_val, 0.0]])
        return model(pt).ravel()[0]

    grad_T = jax.grad(predict_radial)
    hess_T = jax.grad(grad_T)

    r_safe = jnp.where(r == 0.0, 1e-12, r)

    dT_dr = jax.vmap(grad_T)(r)
    d2T_dr2 = jax.vmap(hess_T)(r)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    laplacian = d2T_dr2 + (1.0 / r_safe) * dT_dr
    residual = lambda_vals * laplacian + dlambda_dT * dT_dr**2 + _source(x, source_fn)
    return residual


# ==================== 3D задачи ====================

def laplace_3d(model, x, lam, source_fn=0.0):
    """λ(T)*ΔT + λ'(T)*|∇T|² + f(x,y,z) = 0"""
    def predict(x_pts):
        return model(jnp.atleast_2d(x_pts)).ravel()[0]

    grad_fn = jax.grad(predict)

    def laplacian_pt(x_pt):
        d2x = jax.grad(lambda p: grad_fn(p)[0])(x_pt)
        d2y = jax.grad(lambda p: grad_fn(p)[1])(x_pt)
        d2z = jax.grad(lambda p: grad_fn(p)[2])(x_pt)
        return d2x + d2y + d2z

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    grad_sq = jnp.sum(grad_vals**2, axis=1)
    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual


def cylinder_3d_axisymmetric(model, x, lam, source_fn=0.0):
    """λ(T)*(T_rr + T_r/r + T_zz) + λ'(T)*((T_r)² + (T_z)²) + f(r,z) = 0"""
    def predict(rz):
        return model(rz).ravel()[0]

    grad_fn = jax.grad(predict)

    def laplacian_pt(rz_pt):
        g = grad_fn(rz_pt)
        d2r = jax.grad(lambda p: grad_fn(p)[0])(rz_pt)
        d2z = jax.grad(lambda p: grad_fn(p)[1])(rz_pt)
        r = rz_pt[0]
        r_safe = jnp.where(r == 0.0, 1e-12, r)
        return d2r + (1.0 / r_safe) * g[0] + d2z

    laplacian = jax.vmap(laplacian_pt)(x)
    grad_vals = jax.vmap(grad_fn)(x)

    T_pred = model(x).ravel()
    lambda_vals, dlambda_dT = _lambda_and_derivative(T_pred, lam)

    grad_sq = jnp.sum(grad_vals**2, axis=1)
    residual = lambda_vals * laplacian + dlambda_dT * grad_sq + _source(x, source_fn)
    return residual