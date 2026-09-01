"""
Граничные условия для 2D и 3D задач с нормальными производными.

Поддерживаются:
- Условие Дирихле: T = T_bc на границе
- Условие Неймана: ∂T/∂n = g на границе (нормальная производная)
- Условие Робина: αT + β(∂T/∂n) = h на границе

Ключевая особенность: Нормальная производная ∂T/∂n = ∇T · n вычисляется
с использованием автоматического дифференцирования и предоставленных векторов нормалей.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def dirichlet_bc_2d(model, points: jnp.ndarray, bc_value: float | Callable) -> float:
    """
    Условие Дирихле в 2D: T(x,y) = T_bc

    Args:
        model: Модель нейронной сети
        points: Граничные точки формы (n_points, 2)
        bc_value: Константное значение или функция T_bc(x, y)

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    points_flat = points.reshape(-1, 2)

    if callable(bc_value):
        # bc_value is a function
        predicted = jax.vmap(predict)(points_flat)
        target = jax.vmap(bc_value)(points_flat)
    else:
        # bc_value is constant
        predicted = jax.vmap(predict)(points_flat)
        target = jnp.full_like(predicted, bc_value)

    return jnp.mean((predicted - target) ** 2)


def dirichlet_bc_3d(model, points: jnp.ndarray, bc_value: float | Callable) -> float:
    """
    Условие Дирихле в 3D: T(x,y,z) = T_bc

    Args:
        model: Модель нейронной сети
        points: Граничные точки формы (n_points, 3)
        bc_value: Константное значение или функция T_bc(x, y, z)

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    points_flat = points.reshape(-1, 3)

    if callable(bc_value):
        predicted = jax.vmap(predict)(points_flat)
        target = jax.vmap(bc_value)(points_flat)
    else:
        predicted = jax.vmap(predict)(points_flat)
        target = jnp.full_like(predicted, bc_value)

    return jnp.mean((predicted - target) ** 2)


def neumann_bc_2d(
    model, points: jnp.ndarray, normals: jnp.ndarray, flux_value: float | Callable
) -> float:
    """
    Условие Неймана в 2D: ∂T/∂n = g

    Нормальная производная вычисляется как ∂T/∂n = ∇T · n

    Args:
        model: Модель нейронной сети
        points: Граничные точки формы (n_points, 2)
        normals: Единичные векторы нормалей формы (n_points, 2)
        flux_value: Заданная нормальная производная g (константа или функция)

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    grad_predict = jax.grad(predict)

    def normal_derivative(p, n):
        grad_T = grad_predict(p)
        return jnp.dot(grad_T, n)

    points_flat = points.reshape(-1, 2)
    normals_flat = normals.reshape(-1, 2)

    dT_dn = jax.vmap(normal_derivative)(points_flat, normals_flat)

    if callable(flux_value):
        target = jax.vmap(flux_value)(points_flat)
    else:
        target = jnp.full_like(dT_dn, flux_value)

    return jnp.mean((dT_dn - target) ** 2)


def neumann_bc_3d(
    model, points: jnp.ndarray, normals: jnp.ndarray, flux_value: float | Callable
) -> float:
    """
    Условие Неймана в 3D: ∂T/∂n = g

    Args:
        model: Модель нейронной сети
        points: Граничные точки формы (n_points, 3)
        normals: Единичные векторы нормалей формы (n_points, 3)
        flux_value: Заданная нормальная производная g

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    grad_predict = jax.grad(predict)

    def normal_derivative(p, n):
        grad_T = grad_predict(p)
        return jnp.dot(grad_T, n)

    points_flat = points.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)

    dT_dn = jax.vmap(normal_derivative)(points_flat, normals_flat)

    if callable(flux_value):
        target = jax.vmap(flux_value)(points_flat)
    else:
        target = jnp.full_like(dT_dn, flux_value)

    return jnp.mean((dT_dn - target) ** 2)


def robin_bc_2d(
    model,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    alpha: float,
    beta: float,
    h: float | Callable,
) -> float:
    """
    Условие Робина в 2D: αT + β(∂T/∂n) = h

    Обычная форма для конвективного теплообмена:
    -k(∂T/∂n) = h_conv(T - T_inf)
    что может быть переписано как: h_conv*T + k*(∂T/∂n) = h_conv*T_inf

    Args:
        model: Модель нейронной сети
        points: Граничные точки формы (n_points, 2)
        normals: Единичные векторы нормалей формы (n_points, 2)
        alpha: Коэффициент при члене с T
        beta: Коэффициент при члене с нормальной производной
        h: Значение правой части (константа или функция)

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    grad_predict = jax.grad(predict)

    def normal_derivative(p, n):
        grad_T = grad_predict(p)
        return jnp.dot(grad_T, n)

    points_flat = points.reshape(-1, 2)
    normals_flat = normals.reshape(-1, 2)

    T_vals = jax.vmap(predict)(points_flat)
    dT_dn_vals = jax.vmap(normal_derivative)(points_flat, normals_flat)

    if callable(h):
        h_vals = jax.vmap(h)(points_flat)
    else:
        h_vals = jnp.full_like(T_vals, h)

    residual = alpha * T_vals + beta * dT_dn_vals - h_vals
    return jnp.mean(residual**2)


def robin_bc_3d(
    model,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    alpha: float,
    beta: float,
    h: float | Callable,
) -> float:
    """
    Условие Робина в 3D: αT + β(∂T/∂n) = h

    Args:
        model: Модель нейронной сети
        points: Граничные точки формы (n_points, 3)
        normals: Единичные векторы нормалей формы (n_points, 3)
        alpha: Коэффициент при члене с T
        beta: Коэффициент при члене с нормальной производной
        h: Значение правой части

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    grad_predict = jax.grad(predict)

    def normal_derivative(p, n):
        grad_T = grad_predict(p)
        return jnp.dot(grad_T, n)

    points_flat = points.reshape(-1, 3)
    normals_flat = normals.reshape(-1, 3)

    T_vals = jax.vmap(predict)(points_flat)
    dT_dn_vals = jax.vmap(normal_derivative)(points_flat, normals_flat)

    if callable(h):
        h_vals = jax.vmap(h)(points_flat)
    else:
        h_vals = jnp.full_like(T_vals, h)

    residual = alpha * T_vals + beta * dT_dn_vals - h_vals
    return jnp.mean(residual**2)


def interface_continuity_loss(
    models: tuple,
    interface_points: jnp.ndarray,
    interface_normals: jnp.ndarray,
    lambdas: tuple,
) -> tuple[float, float]:
    """
    Условия сопряжения между двумя доменами для многодоменной ФИНС.

    Обеспечивает:
    1. Непрерывность температуры: T₁ = T₂
    2. Непрерывность потока: λ₁(∂T₁/∂n) = λ₂(∂T₂/∂n)

    На основе формулировок XPINN/cPINN (Jagtap & Karniadakis, 2020).

    Args:
        models: Кортеж из двух нейронных сетей (model_1, model_2)
        interface_points: Точки на интерфейсе формы (n_points, dim)
        interface_normals: Единичные нормали, направленные от домена 1 к домену 2
        lambdas: Кортеж коэффициентов теплопроводности (lambda_1, lambda_2)

    Returns:
        Кортеж (потеря_непрерывности_температуры, потеря_непрерывности_потока)
    """
    model_1, model_2 = models
    lambda_1, lambda_2 = lambdas

    def predict_1(p):
        return model_1(jnp.atleast_2d(p)).ravel()[0]

    def predict_2(p):
        return model_2(jnp.atleast_2d(p)).ravel()[0]

    grad_1 = jax.grad(predict_1)
    grad_2 = jax.grad(predict_2)

    def compute_losses(p, n):
        # Temperature values
        T_1 = predict_1(p)
        T_2 = predict_2(p)

        # Normal derivatives
        grad_T_1 = grad_1(p)
        grad_T_2 = grad_2(p)

        dT1_dn = jnp.dot(grad_T_1, n)
        dT2_dn = jnp.dot(grad_T_2, n)

        # Continuity residuals
        temp_residual = (T_1 - T_2) ** 2
        flux_residual = (lambda_1 * dT1_dn - lambda_2 * dT2_dn) ** 2

        return temp_residual, flux_residual

    points_flat = interface_points.reshape(-1, interface_points.shape[1])
    normals_flat = interface_normals.reshape(-1, interface_normals.shape[1])

    losses = jax.vmap(compute_losses)(points_flat, normals_flat)

    temp_loss = jnp.mean(losses[0])
    flux_loss = jnp.mean(losses[1])

    return temp_loss, flux_loss


def interface_continuity_loss_nd(
    models: tuple, interface_data: dict, all_lambdas: tuple, weights: tuple = (1.0, 1.0)
) -> float:
    """
    Обобщенная потеря на интерфейсе для нескольких интерфейсов в N измерениях.

    Формулировка в стиле XPINN с поддержкой произвольного количества интерфейсов.

    Args:
        models: Кортеж нейронных сетей для каждого домена
        interface_data: Список словарей с ключами:
            - 'points': точки коллокации на интерфейсе
            - 'normals': векторы нормалей
            - 'domains': кортеж индексов доменов (i, j), разделяющих этот интерфейс
        all_lambdas: Кортеж значений теплопроводности для каждого домена
        weights: Веса для непрерывности (температура, поток)

    Returns:
        Полная потеря на интерфейсе
    """
    total_loss = 0.0
    w_temp, w_flux = weights

    for iface in interface_data:
        points = iface["points"]
        normals = iface["normals"]
        dom_i, dom_j = iface["domains"]

        model_i = models[dom_i]
        model_j = models[dom_j]
        lambda_i = all_lambdas[dom_i]
        lambda_j = all_lambdas[dom_j]

        # Compute interface losses
        temp_loss, flux_loss = interface_continuity_loss(
            (model_i, model_j), points, normals, (lambda_i, lambda_j)
        )

        total_loss += w_temp * temp_loss + w_flux * flux_loss

    return total_loss


def convective_bc_2d(
    model,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h_conv: float,
    T_inf: float,
    k: float,
) -> float:
    """
    Конвективное граничное условие (тип Робина) в 2D:
    -k(∂T/∂n) = h_conv(T - T_inf)

    В переписанном виде: h_conv*T + k*(∂T/∂n) = h_conv*T_inf

    Args:
        model: Модель нейронной сети
        points: Граничные точки
        normals: Внешние единичные нормали
        h_conv: Коэффициент конвективной теплоотдачи
        T_inf: Температура окружающей среды
        k: Теплопроводность твердого тела

    Returns:
        Среднеквадратичная невязка
    """
    return robin_bc_2d(model, points, normals, alpha=h_conv, beta=k, h=h_conv * T_inf)


def convective_bc_3d(
    model,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h_conv: float,
    T_inf: float,
    k: float,
) -> float:
    """
    Конвективное граничное условие в 3D.

    Args:
        model: Модель нейронной сети
        points: Граничные точки
        normals: Внешние единичные нормали
        h_conv: Коэффициент конвективной теплоотдачи
        T_inf: Температура окружающей среды
        k: Теплопроводность

    Returns:
        Среднеквадратичная невязка
    """
    return robin_bc_3d(model, points, normals, alpha=h_conv, beta=k, h=h_conv * T_inf)


def radiation_bc_2d(
    model,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    epsilon: float,
    sigma: float,
    T_surround: float,
    k: float,
) -> float:
    """
    Радиационное граничное условие (нелинейный тип Робина) в 2D:
    -k(∂T/∂n) = εσ(T⁴ - T_surround⁴)

    Это нелинейное условие по T и требует специальной обработки.

    Args:
        model: Модель нейронной сети
        points: Граничные точки
        normals: Внешние единичные нормали
        epsilon: Коэффициент излучения (чернота)
        sigma: Постоянная Стефана-Больцмана
        T_surround: Температура окружающей среды
        k: Теплопроводность

    Returns:
        Среднеквадратичная невязка
    """

    def predict(p):
        return model(jnp.atleast_2d(p)).ravel()[0]

    grad_predict = jax.grad(predict)

    def radiative_residual(p, n):
        T = predict(p)
        grad_T = grad_predict(p)
        dT_dn = jnp.dot(grad_T, n)

        # -k*dT/dn = epsilon*sigma*(T^4 - T_surround^4)
        lhs = -k * dT_dn
        rhs = epsilon * sigma * (T**4 - T_surround**4)

        return lhs - rhs

    points_flat = points.reshape(-1, 2)
    normals_flat = normals.reshape(-1, 2)

    residuals = jax.vmap(radiative_residual)(points_flat, normals_flat)
    return jnp.mean(residuals**2)


# Convenience aliases
dirichlet_2d = dirichlet_bc_2d
dirichlet_3d = dirichlet_bc_3d
neumann_2d = neumann_bc_2d
neumann_3d = neumann_bc_3d
robin_2d = robin_bc_2d
robin_3d = robin_bc_3d
convective_2d = convective_bc_2d
convective_3d = convective_bc_3d
interface_loss = interface_continuity_loss
