"""
Модуль источников тепла для стационарных задач теплопроводности.

Источники представляются как функции f(x), где x - пространственная координата.
Уравнение теплопроводности с источником: -d/dx(k*dT/dx) = f(x)
"""

import jax.numpy as jnp
from jax import Array
from typing import Callable


def constant_source(value: float) -> Callable[[Array], Array]:
    """
    Создает функцию постоянного источника тепла f(x) = value.
    
    Args:
        value: Значение константы (Вт/м³).
        
    Returns:
        Функция источника, принимающая x и возвращающая f(x).
    """
    def _source(x: Array) -> Array:
        return jnp.full_like(x, value)
    return _source


def gaussian_source(center: float, width: float, amplitude: float) -> Callable[[Array], Array]:
    """
    Создает функцию гауссова источника тепла.
    f(x) = amplitude * exp(-(x - center)² / (2 * width²))
    
    Args:
        center: Центр источника (м).
        width: Ширина (sigma) источника (м).
        amplitude: Амплитуда источника (Вт/м³).
        
    Returns:
        Функция источника.
    """
    def _source(x: Array) -> Array:
        return amplitude * jnp.exp(-((x - center) ** 2) / (2 * width ** 2))
    return _source


def point_source(location: float, epsilon: float = 0.01, amplitude: float = 1.0) -> Callable[[Array], Array]:
    """
    Аппроксимирует точечный источник (дельта-функцию) узким гауссианом.
    
    Args:
        location: Положение точечного источника (м).
        epsilon: Ширина аппроксимации (чем меньше, тем ближе к дельте).
        amplitude: Интегральная мощность источника (Вт).
        
    Returns:
        Функция источника.
    """
    # Нормировка чтобы интеграл был равен amplitude
    norm = amplitude / (jnp.sqrt(2 * jnp.pi) * epsilon)
    
    def _source(x: Array) -> Array:
        return norm * jnp.exp(-((x - location) ** 2) / (2 * epsilon ** 2))
    return _source


def custom_source(func: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """
    Обертка для пользовательской функции источника.
    
    Args:
        func: Пользовательская функция f(x).
        
    Returns:
        Та же функция.
    """
    return func


def no_source() -> Callable[[Array], Array]:
    """
    Возвращает функцию нулевого источника (однородное уравнение).
    
    Returns:
        Функция, всегда возвращающая 0.
    """
    def _source(x: Array) -> Array:
        return jnp.zeros_like(x)
    return _source
