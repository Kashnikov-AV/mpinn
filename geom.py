import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod


class Geometry(ABC):
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def sample_interior(self, n_points, method='random', rng=None):
        pass

    @abstractmethod
    def sample_boundary(self):
        pass


class Interval(Geometry):

    def __init__(self, x_left, x_right):
        super().__init__(dim=1)
        self.x_left = float(x_left)
        self.x_right = float(x_right)

    def sample_interior(self, n_points, method='random', rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        if method == 'random':
            return jax.random.uniform(
                rng, shape=(n_points, 1), minval=self.x_left, maxval=self.x_right
            )
        if method == 'uniform':
            return jnp.linspace(self.x_left, self.x_right, n_points).reshape(-1, 1)
        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self):
        return jnp.array([[self.x_left], [self.x_right]])

    def generate_collocation(self, n_interior=100, method='random', rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method, keys[0])
        boundary = self.sample_boundary()
        
        # Склеиваем внутренние и граничные точки в один массив
        return jnp.vstack([interior, boundary])


class Rectangle(Geometry):
    """
    Двумерный прямоугольник x_min, x_max × y_min, y_max
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        """
        Args:
            x_min, x_max: границы по оси x.
            y_min, y_max: границы по оси y.
        """
        super().__init__(dim=2)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def sample_interior(self, n_points, method='random', rng=None):
        """
        Генерирует точки внутри прямоугольника.

        Args:
            n_points: количество точек
            method: 'random' – случайная выборка, 'uniform'
            rng: ключ PRNG.

        Returns:
            Массив формы (n_points, 2).
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == 'random':
            keys = jax.random.split(rng, 2)
            x = jax.random.uniform(keys[0], (n_points, 1), minval=self.x_min, maxval=self.x_max)
            y = jax.random.uniform(keys[1], (n_points, 1), minval=self.y_min, maxval=self.y_max)
            return jnp.hstack([x, y])
        if method == 'uniform':
            # Генерируем равномерную сетку (приближённо)
            n_per_side = int(jnp.sqrt(n_points))
            x = jnp.linspace(self.x_min, self.x_max, n_per_side)
            y = jnp.linspace(self.y_min, self.y_max, n_per_side)
            xx, yy = jnp.meshgrid(x, y, indexing='ij')
            points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
            return points[:n_points]  # обрезаем до нужного количества
        raise ValueError(f'Unknown method: {method}')

    def sample_boundary(self, n_points=None, method='random', rng=None):
        """
        Генерирует точки на границе (периметре) прямоугольника.

        Если n_points не указан, возвращает четыре угловые точки.

        Args:
            n_points: общее количество точек на границе (распределяются равномерно по сторонам).
            method: 'random' – случайное расположение на периметре.
            rng: ключ PRNG.

        Returns:
            Массив формы (n_points, 2) или (4, 2) если n_points=None.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if n_points is None:
            # Возвращаем четыре угла
            return jnp.array([
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max]
            ])

        if method == 'random':
            # Генерируем точки на периметре с равномерным распределением по длине
            # Вычисляем длины сторон
            dx = self.x_max - self.x_min
            dy = self.y_max - self.y_min
            perimeter = 2 * (dx + dy)

            # Случайные смещения по периметру
            s = jax.random.uniform(rng, (n_points, 1), minval=0.0, maxval=perimeter)

            # Определяем, на какой стороне находится точка
            # Используем кумулятивные длины: 0..dx (нижняя), dx..dx+dy (правая), dx+dy..2dx+dy (верхняя), 2dx+dy..perimeter (левая)
            points = jnp.zeros((n_points, 2))
            # Нижняя сторона (y = y_min)
            mask1 = s <= dx
            t = s[mask1] / dx
            points = points.at[mask1, 0].set(self.x_min + t * dx)
            points = points.at[mask1, 1].set(self.y_min)

            # Правая сторона (x = x_max)
            mask2 = (s > dx) & (s <= dx + dy)
            t = (s[mask2] - dx) / dy
            points = points.at[mask2, 0].set(self.x_max)
            points = points.at[mask2, 1].set(self.y_min + t * dy)

            # Верхняя сторона (y = y_max)
            mask3 = (s > dx + dy) & (s <= 2 * dx + dy)
            t = (s[mask3] - dx - dy) / dx
            points = points.at[mask3, 0].set(self.x_max - t * dx)
            points = points.at[mask3, 1].set(self.y_max)

            # Левая сторона (x = x_min)
            mask4 = s > 2 * dx + dy
            t = (s[mask4] - 2 * dx - dy) / dy
            points = points.at[mask4, 0].set(self.x_min)
            points = points.at[mask4, 1].set(self.y_max - t * dy)

            return points

        raise ValueError(f'Unknown method: {method}')

    def generate_collocation(self, n_interior=100, n_boundary=None,
                             method_interior='random', method_boundary='random',
                             rng=None):
        """
        Генерирует набор точек для коллокации: внутренние + граничные.

        Args:
            n_interior: количество внутренних точек.
            n_boundary: количество граничных точек (если None, то возвращаются только углы).
            method_interior: метод для внутренних точек.
            method_boundary: метод для граничных точек.
            rng: ключ PRNG.

        Returns:
            Массив точек формы (n_interior + n_boundary, 2) или (n_interior + 4, 2).
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])