import jax
import jax.numpy as jnp

def dirichlet_bc(model, x, T):
    return (model(jnp.array([[x]])).ravel()[0] - T) ** 2

def neuman_bc(model, x, g):
    def get_T(x_val):
        return model(jnp.array([[x_val]])).ravel()[0]
        
    dT_dx = jax.grad(get_T)(x)
    return (dT_dx - g) ** 2

def robin_bc(model, x, alpha, beta, h):
    def get_T(x_val):
        return model(jnp.array([[x_val]])).ravel()[0]
        
    T_val = get_T(x)
    dT_dx = jax.grad(get_T)(x)
    return (alpha * T_val + beta * dT_dx - h) ** 2