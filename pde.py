import jax
import jax.numpy as jnp

def cylinder_1d(model, x, phys):
    def T(r_val):
        return model(jnp.array([[r_val]])).ravel()[0]
    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-8, r)
    dT_dr = jax.vmap(jax.grad(T))(r)
    d2T_dr2 = jax.vmap(jax.grad(jax.grad(T)))(r)
    return jnp.mean((dT_dr / r_safe + d2T_dr2) ** 2)

def line_1d(model, x, phys):
    def y(x_val):
        return model(jnp.array([[x_val]])).ravel()[0]
    d2y_dx2 = jax.vmap(jax.grad(jax.grad(y)))(x.ravel())
    return jnp.mean(d2y_dx2 ** 2)

def sphere_1d(model, x, phys):
    def y(r_val):
        return model(jnp.array([[r_val]])).ravel()[0]
    r = x.ravel()
    r_safe = jnp.where(r == 0.0, 1e-8, r)
    dy_dr = jax.vmap(jax.grad(y))(r)
    d2y_dr2 = jax.vmap(jax.grad(jax.grad(y)))(r)
    return jnp.mean((d2y_dr2 + (2.0 / r_safe) * dy_dr) ** 2)