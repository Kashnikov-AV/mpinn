import jax
import jax.numpy as jnp

def dirichlet_bc(model, x, T):
    """
    Вычисляет среднюю квадратичную невязку граничного условия Дирихле.
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Координаты граничных точек.
    T : float или jax.Array
        Заданное значение температуры на границе.
    
    Returns:
    --------
    float
        Средняя квадратичная невязка (1/N * sum((T_pred - T)^2)).
    """
    residuals = jnp.array([(model(jnp.array([[xi]])).ravel()[0] - T) ** 2 for xi in x])
    return jnp.mean(residuals)

def neuman_bc(model, x, g):
    """
    Вычисляет среднюю квадратичную невязку граничного условия Неймана.
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Координаты граничных точек.
    g : float или jax.Array
        Заданное значение градиента температуры на границе.
    
    Returns:
    --------
    float
        Средняя квадратичная невязка (1/N * sum((dT/dx - g)^2)).
    """
    def get_T(x_val):
        return model(jnp.array([[x_val]])).ravel()[0]
        
    def compute_residual(xi):
        dT_dx = jax.grad(get_T)(xi)
        return (dT_dx - g) ** 2
    
    residuals = jnp.array([compute_residual(xi) for xi in x])
    return jnp.mean(residuals)

def robin_bc(model, x, alpha, beta, h):
    """
    Вычисляет среднюю квадратичную невязку граничного условия Робина.
    
    Parameters:
    -----------
    model : nnx.Module
        Нейронная сеть.
    x : jax.Array
        Координаты граничных точек.
    alpha : float
        Коэффициент при температуре.
    beta : float
        Коэффициент при градиенте температуры.
    h : float
        Заданное значение комбинации.
    
    Returns:
    --------
    float
        Средняя квадратичная невязка (1/N * sum((alpha*T + beta*dT/dx - h)^2)).
    """
    def get_T(x_val):
        return model(jnp.array([[x_val]])).ravel()[0]
        
    def compute_residual(xi):
        T_val = get_T(xi)
        dT_dx = jax.grad(get_T)(xi)
        return (alpha * T_val + beta * dT_dx - h) ** 2
    
    residuals = jnp.array([compute_residual(xi) for xi in x])
    return jnp.mean(residuals)