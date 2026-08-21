"""
PDE operators for 2D and 3D problems.

Provides vectorized PDE residual functions for:
- Laplace equation (steady heat conduction)
- Poisson equation (with source term)
- Heat equation (transient, using automatic differentiation in time)
- Convection-diffusion equation

All functions work with JAX automatic differentiation and support
arbitrary spatial dimensions (2D/3D).
"""

import jax
import jax.numpy as jnp
from typing import Optional, Callable, Dict, Any
from functools import partial


def laplacian_2d(model, points: jnp.ndarray, phys: Optional[Any] = None) -> float:
    """
    Compute Laplacian residual in 2D: ∇²T = ∂²T/∂x² + ∂²T/∂y²
    
    For steady heat conduction without source: ∇²T = 0
    With source: ∇²T + f(x,y) = 0
    
    Args:
        model: Neural network model
        points: Collocation points of shape (n_points, 2)
        phys: Optional physics parameters with source_fn
        
    Returns:
        Mean squared residual
    """
    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]
    
    # Compute Hessian using grad twice
    grad_predict = jax.grad(predict)
    
    def d2T_dx2(p):
        def dT_dx(x_val):
            return predict(jnp.array([x_val, p[1]]))
        return jax.grad(dT_dx)(p[0])
    
    def d2T_dy2(p):
        def dT_dy(y_val):
            return predict(jnp.array([p[0], y_val]))
        return jax.grad(dT_dy)(p[1])
    
    # Vectorize over all points
    points_flat = points.reshape(-1, 2)
    d2T_dx2_vals = jax.vmap(d2T_dx2)(points_flat)
    d2T_dy2_vals = jax.vmap(d2T_dy2)(points_flat)
    
    laplacian = d2T_dx2_vals + d2T_dy2_vals
    
    # Add source term if provided
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_vals = jax.vmap(lambda p: phys.source_fn(p))(points_flat)
        laplacian = laplacian + source_vals
    
    return jnp.mean(laplacian ** 2)


def laplacian_3d(model, points: jnp.ndarray, phys: Optional[Any] = None) -> float:
    """
    Compute Laplacian residual in 3D: ∇²T = ∂²T/∂x² + ∂²T/∂y² + ∂²T/∂z²
    
    Args:
        model: Neural network model
        points: Collocation points of shape (n_points, 3)
        phys: Optional physics parameters with source_fn
        
    Returns:
        Mean squared residual
    """
    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]
    
    def d2T_dx2(p):
        def dT_dx(x_val):
            return predict(jnp.array([x_val, p[1], p[2]]))
        return jax.grad(dT_dx)(p[0])
    
    def d2T_dy2(p):
        def dT_dy(y_val):
            return predict(jnp.array([p[0], y_val, p[2]]))
        return jax.grad(dT_dy)(p[1])
    
    def d2T_dz2(p):
        def dT_dz(z_val):
            return predict(jnp.array([p[0], p[1], z_val]))
        return jax.grad(dT_dz)(p[2])
    
    points_flat = points.reshape(-1, 3)
    d2T_dx2_vals = jax.vmap(d2T_dx2)(points_flat)
    d2T_dy2_vals = jax.vmap(d2T_dy2)(points_flat)
    d2T_dz2_vals = jax.vmap(d2T_dz2)(points_flat)
    
    laplacian = d2T_dx2_vals + d2T_dy2_vals + d2T_dz2_vals
    
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_vals = jax.vmap(lambda p: phys.source_fn(p))(points_flat)
        laplacian = laplacian + source_vals
    
    return jnp.mean(laplacian ** 2)


def laplacian_nd(model, points: jnp.ndarray, phys: Optional[Any] = None) -> float:
    """
    Compute Laplacian residual in arbitrary dimension using trace of Hessian.
    
    More efficient than component-wise computation for higher dimensions.
    
    Args:
        model: Neural network model
        points: Collocation points of shape (n_points, dim)
        phys: Optional physics parameters with source_fn
        
    Returns:
        Mean squared residual
    """
    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]
    
    # Use jacobian of gradient to get Hessian
    grad_predict = jax.grad(predict)
    hessian_predict = jax.jacfwd(grad_predict)
    
    points_flat = points.reshape(-1, points.shape[1])
    
    # Compute trace of Hessian (sum of diagonal elements)
    def laplacian_at_point(p):
        hess = hessian_predict(p)
        return jnp.trace(hess)
    
    laplacians = jax.vmap(laplacian_at_point)(points_flat)
    
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_vals = jax.vmap(lambda p: phys.source_fn(p))(points_flat)
        laplacians = laplacians + source_vals
    
    return jnp.mean(laplacians ** 2)


def heat_equation_2d(model, points_with_time: jnp.ndarray, 
                     phys: Optional[Any] = None) -> float:
    """
    Transient heat equation in 2D: ∂T/∂t = α∇²T + Q
    
    where α is thermal diffusivity and Q is source term.
    
    Args:
        model: Neural network model taking (x, y, t) as input
        points_with_time: Points of shape (n_points, 3) = (x, y, t)
        phys: Physics parameters with alpha (diffusivity) and source_fn
        
    Returns:
        Mean squared residual
    """
    def predict(pt):
        return model(jnp.atleast_2d(pt)).ravel()[0]
    
    # Time derivative
    def dT_dt(pt):
        def T_at_t(t_val):
            return predict(jnp.array([pt[0], pt[1], t_val]))
        return jax.grad(T_at_t)(pt[2])
    
    # Spatial Laplacian
    def d2T_dx2(pt):
        def dT_dx(x_val):
            return predict(jnp.array([x_val, pt[1], pt[2]]))
        return jax.grad(dT_dx)(pt[0])
    
    def d2T_dy2(pt):
        def dT_dy(y_val):
            return predict(jnp.array([pt[0], y_val, pt[2]]))
        return jax.grad(dT_dy)(pt[1])
    
    points_flat = points_with_time.reshape(-1, 3)
    
    dT_dt_vals = jax.vmap(dT_dt)(points_flat)
    d2T_dx2_vals = jax.vmap(d2T_dx2)(points_flat)
    d2T_dy2_vals = jax.vmap(d2T_dy2)(points_flat)
    
    alpha = getattr(phys, 'alpha', 1.0)  # Thermal diffusivity
    laplacian = d2T_dx2_vals + d2T_dy2_vals
    
    residual = dT_dt_vals - alpha * laplacian
    
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_vals = jax.vmap(lambda p: phys.source_fn(p))(points_flat)
        residual = residual - source_vals
    
    return jnp.mean(residual ** 2)


def heat_equation_3d(model, points_with_time: jnp.ndarray,
                     phys: Optional[Any] = None) -> float:
    """
    Transient heat equation in 3D: ∂T/∂t = α∇²T + Q
    
    Args:
        model: Neural network model taking (x, y, z, t) as input
        points_with_time: Points of shape (n_points, 4) = (x, y, z, t)
        phys: Physics parameters with alpha and source_fn
        
    Returns:
        Mean squared residual
    """
    def predict(pt):
        return model(jnp.atleast_2d(pt)).ravel()[0]
    
    def dT_dt(pt):
        def T_at_t(t_val):
            return predict(jnp.array([pt[0], pt[1], pt[2], t_val]))
        return jax.grad(T_at_t)(pt[3])
    
    def d2T_dx2(pt):
        def dT_dx(x_val):
            return predict(jnp.array([x_val, pt[1], pt[2], pt[3]]))
        return jax.grad(dT_dx)(pt[0])
    
    def d2T_dy2(pt):
        def dT_dy(y_val):
            return predict(jnp.array([pt[0], y_val, pt[2], pt[3]]))
        return jax.grad(dT_dy)(pt[1])
    
    def d2T_dz2(pt):
        def dT_dz(z_val):
            return predict(jnp.array([pt[0], pt[1], z_val, pt[3]]))
        return jax.grad(dT_dz)(pt[2])
    
    points_flat = points_with_time.reshape(-1, 4)
    
    dT_dt_vals = jax.vmap(dT_dt)(points_flat)
    d2T_dx2_vals = jax.vmap(d2T_dx2)(points_flat)
    d2T_dy2_vals = jax.vmap(d2T_dy2)(points_flat)
    d2T_dz2_vals = jax.vmap(d2T_dz2)(points_flat)
    
    alpha = getattr(phys, 'alpha', 1.0)
    laplacian = d2T_dx2_vals + d2T_dy2_vals + d2T_dz2_vals
    
    residual = dT_dt_vals - alpha * laplacian
    
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_vals = jax.vmap(lambda p: phys.source_fn(p))(points_flat)
        residual = residual - source_vals
    
    return jnp.mean(residual ** 2)


def convection_diffusion_2d(model, points: jnp.ndarray,
                            phys: Optional[Any] = None) -> float:
    """
    Steady convection-diffusion equation in 2D:
    u·∇T = D∇²T + S
    
    where u is velocity field, D is diffusion coefficient, S is source.
    
    Args:
        model: Neural network model
        points: Collocation points of shape (n_points, 2)
        phys: Physics parameters with velocity_fn, diffusion_coef, source_fn
        
    Returns:
        Mean squared residual
    """
    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]
    
    grad_predict = jax.grad(predict)
    
    def gradient_T(p):
        return grad_predict(p)
    
    def laplacian_T(p):
        hess = jax.jacfwd(grad_predict)(p)
        return jnp.trace(hess)
    
    points_flat = points.reshape(-1, 2)
    
    grads = jax.vmap(gradient_T)(points_flat)
    laplacians = jax.vmap(laplacian_T)(points_flat)
    
    # Get velocity field
    if hasattr(phys, 'velocity_fn') and phys.velocity_fn is not None:
        velocities = jax.vmap(lambda p: phys.velocity_fn(p))(points_flat)
    else:
        velocities = jnp.zeros_like(grads)
    
    diffusion_coef = getattr(phys, 'diffusion_coef', 1.0)
    
    # u·∇T - D∇²T - S = 0
    convection = jax.vmap(lambda u, g: jnp.dot(u, g))(velocities, grads)
    diffusion = diffusion_coef * laplacians
    
    residual = convection - diffusion
    
    if hasattr(phys, 'source_fn') and phys.source_fn is not None:
        source_vals = jax.vmap(lambda p: phys.source_fn(p))(points_flat)
        residual = residual - source_vals
    
    return jnp.mean(residual ** 2)


def poisson_equation_nd(model, points: jnp.ndarray, source_fn: Callable) -> float:
    """
    Poisson equation in arbitrary dimension: ∇²T = f(x)
    
    Args:
        model: Neural network model
        points: Collocation points of shape (n_points, dim)
        source_fn: Function computing source term f(x)
        
    Returns:
        Mean squared residual
    """
    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]
    
    grad_predict = jax.grad(predict)
    hessian_predict = jax.jacfwd(grad_predict)
    
    points_flat = points.reshape(-1, points.shape[1])
    
    def poisson_residual(p):
        hess = hessian_predict(p)
        laplacian = jnp.trace(hess)
        source = source_fn(p)
        return laplacian - source
    
    residuals = jax.vmap(poisson_residual)(points_flat)
    return jnp.mean(residuals ** 2)


def variable_coefficient_laplacian(model, points: jnp.ndarray,
                                   conductivity_fn: Callable) -> float:
    """
    Variable coefficient Laplacian: ∇·(k(x)∇T) = 0
    
    For heat conduction with spatially varying thermal conductivity k(x).
    
    Expands to: k(x)∇²T + ∇k·∇T = 0
    
    Args:
        model: Neural network model
        points: Collocation points of shape (n_points, dim)
        conductivity_fn: Function computing k(x) and optionally ∇k
        
    Returns:
        Mean squared residual
    """
    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]
    
    grad_predict = jax.grad(predict)
    hessian_predict = jax.jacfwd(grad_predict)
    
    points_flat = points.reshape(-1, points.shape[1])
    dim = points.shape[1]
    
    def residual_at_point(p):
        grad_T = grad_predict(p)
        hess_T = hessian_predict(p)
        laplacian_T = jnp.trace(hess_T)
        
        # Get conductivity and its gradient
        k_result = conductivity_fn(p)
        if isinstance(k_result, tuple):
            k, grad_k = k_result
        else:
            k = k_result
            # Compute grad_k numerically
            def k_fn(x):
                return conductivity_fn(jnp.atleast_1d(x))
            grad_k = jax.grad(k_fn)(p)
        
        # ∇·(k∇T) = k∇²T + ∇k·∇T
        return k * laplacian_T + jnp.dot(grad_k, grad_T)
    
    residuals = jax.vmap(residual_at_point)(points_flat)
    return jnp.mean(residuals ** 2)


# Convenience aliases
laplace_2d = laplacian_2d
laplace_3d = laplacian_3d
laplace_nd = laplacian_nd
heat_2d = heat_equation_2d
heat_3d = heat_equation_3d
conv_diff_2d = convection_diffusion_2d
