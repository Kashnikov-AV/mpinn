import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Geometry(ABC):
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def sample_interior(self, n_points, method='random', rng=None):
        pass

    @abstractmethod
    def sample_boundary(self):
        pass

    def plot_domain(self, interior_points, boundary_points=None, interface_points=None, title="Domain Visualization"):
        """
        Визуализирует домен с точками коллокации.
        
        Args:
            interior_points: внутренние точки (N, dim)
            boundary_points: граничные точки (M, dim), опционально
            interface_points: точки на интерфейсах доменов (K, dim), опционально
            title: заголовок графика
        """
        if self.dim == 2:
            self._plot_2d(interior_points, boundary_points, interface_points, title)
        elif self.dim == 3:
            self._plot_3d(interior_points, boundary_points, interface_points, title)
        else:
            raise ValueError(f"Visualization not supported for {self.dim}D")

    def _plot_2d(self, interior, boundary=None, interface=None, title="Domain Visualization"):
        """Отрисовка 2D домена"""
        plt.figure(figsize=(8, 6))
        
        # Внутренние точки - синие
        if interior is not None and len(interior) > 0:
            plt.scatter(interior[:, 0], interior[:, 1], c='blue', s=10, alpha=0.5, label='Interior')
        
        # Граничные точки - зеленые
        if boundary is not None and len(boundary) > 0:
            plt.scatter(boundary[:, 0], boundary[:, 1], c='green', s=20, alpha=0.7, label='Boundary')
        
        # Точки на интерфейсах - красные
        if interface is not None and len(interface) > 0:
            plt.scatter(interface[:, 0], interface[:, 1], c='red', s=20, alpha=0.7, label='Interface')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def _plot_3d(self, interior, boundary=None, interface=None, title="Domain Visualization"):
        """Отрисовка 3D домена"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Внутренние точки - синие
        if interior is not None and len(interior) > 0:
            ax.scatter(interior[:, 0], interior[:, 1], interior[:, 2], c='blue', s=10, alpha=0.5, label='Interior')
        
        # Граничные точки - зеленые
        if boundary is not None and len(boundary) > 0:
            ax.scatter(boundary[:, 0], boundary[:, 1], boundary[:, 2], c='green', s=20, alpha=0.7, label='Boundary')
        
        # Точки на интерфейсах - красные
        if interface is not None and len(interface) > 0:
            ax.scatter(interface[:, 0], interface[:, 1], interface[:, 2], c='red', s=20, alpha=0.7, label='Interface')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


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


class Circle(Geometry):
    """
    Двумерный круг с центром в (cx, cy) и радиусом R
    """

    def __init__(self, cx=0.0, cy=0.0, radius=1.0):
        super().__init__(dim=2)
        self.cx = float(cx)
        self.cy = float(cy)
        self.radius = float(radius)

    def sample_interior(self, n_points, method='random', rng=None):
        """Генерирует точки внутри круга"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == 'random':
            keys = jax.random.split(rng, 2)
            # Равномерное распределение в круге через полярные координаты
            r = self.radius * jnp.sqrt(jax.random.uniform(keys[0], (n_points, 1)))
            theta = 2 * jnp.pi * jax.random.uniform(keys[1], (n_points, 1))
            x = self.cx + r * jnp.cos(theta)
            y = self.cy + r * jnp.sin(theta)
            return jnp.hstack([x, y])
        
        raise ValueError(f'Unknown method: {method}')

    def sample_boundary(self, n_points=100, method='random', rng=None):
        """Генерирует точки на окружности"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == 'random':
            theta = 2 * jnp.pi * jax.random.uniform(rng, (n_points, 1))
            x = self.cx + self.radius * jnp.cos(theta)
            y = self.cy + self.radius * jnp.sin(theta)
            return jnp.hstack([x, y])
        
        raise ValueError(f'Unknown method: {method}')

    def generate_collocation(self, n_interior=100, n_boundary=50,
                             method_interior='random', method_boundary='random',
                             rng=None):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])


class Box(Geometry):
    """
    Трехмерный параллелепипед x_min..x_max × y_min..y_max × z_min..z_max
    """

    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        super().__init__(dim=3)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def sample_interior(self, n_points, method='random', rng=None):
        """Генерирует точки внутри параллелепипеда"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == 'random':
            keys = jax.random.split(rng, 3)
            x = jax.random.uniform(keys[0], (n_points, 1), minval=self.x_min, maxval=self.x_max)
            y = jax.random.uniform(keys[1], (n_points, 1), minval=self.y_min, maxval=self.y_max)
            z = jax.random.uniform(keys[2], (n_points, 1), minval=self.z_min, maxval=self.z_max)
            return jnp.hstack([x, y, z])
        
        raise ValueError(f'Unknown method: {method}')

    def sample_boundary(self, n_points=None, method='random', rng=None):
        """Генерирует точки на границе параллелепипеда"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if n_points is None:
            # Возвращаем 8 углов
            corners = []
            for x in [self.x_min, self.x_max]:
                for y in [self.y_min, self.y_max]:
                    for z in [self.z_min, self.z_max]:
                        corners.append([x, y, z])
            return jnp.array(corners)

        if method == 'random':
            # Генерируем точки на 6 гранях
            dx = self.x_max - self.x_min
            dy = self.y_max - self.y_min
            dz = self.z_max - self.z_min
            
            # Площадь граней
            areas = [
                dx * dy,  # bottom (z=z_min)
                dx * dy,  # top (z=z_max)
                dx * dz,  # front (y=y_min)
                dx * dz,  # back (y=y_max)
                dy * dz,  # left (x=x_min)
                dy * dz,  # right (x=x_max)
            ]
            total_area = sum(areas)
            
            # Распределяем точки пропорционально площадям
            points_per_face = [int(n_points * a / total_area) for a in areas]
            points_per_face[-1] += n_points - sum(points_per_face)  # корректировка
            
            all_points = []
            keys = jax.random.split(rng, 6)
            
            # Bottom (z=z_min)
            if points_per_face[0] > 0:
                x = jax.random.uniform(keys[0], (points_per_face[0], 1), minval=self.x_min, maxval=self.x_max)
                y = jax.random.uniform(keys[1], (points_per_face[0], 1), minval=self.y_min, maxval=self.y_max)
                z = jnp.full((points_per_face[0], 1), self.z_min)
                all_points.append(jnp.hstack([x, y, z]))
            
            # Top (z=z_max)
            if points_per_face[1] > 0:
                x = jax.random.uniform(keys[2], (points_per_face[1], 1), minval=self.x_min, maxval=self.x_max)
                y = jax.random.uniform(keys[3], (points_per_face[1], 1), minval=self.y_min, maxval=self.y_max)
                z = jnp.full((points_per_face[1], 1), self.z_max)
                all_points.append(jnp.hstack([x, y, z]))
            
            # Front (y=y_min)
            if points_per_face[2] > 0:
                x = jax.random.uniform(keys[4], (points_per_face[2], 1), minval=self.x_min, maxval=self.x_max)
                z = jax.random.uniform(keys[5], (points_per_face[2], 1), minval=self.z_min, maxval=self.z_max)
                y = jnp.full((points_per_face[2], 1), self.y_min)
                all_points.append(jnp.hstack([x, y, z]))
            
            # Back (y=y_max)
            if points_per_face[3] > 0:
                x = jax.random.uniform(keys[0], (points_per_face[3], 1), minval=self.x_min, maxval=self.x_max)
                z = jax.random.uniform(keys[1], (points_per_face[3], 1), minval=self.z_min, maxval=self.z_max)
                y = jnp.full((points_per_face[3], 1), self.y_max)
                all_points.append(jnp.hstack([x, y, z]))
            
            # Left (x=x_min)
            if points_per_face[4] > 0:
                y = jax.random.uniform(keys[2], (points_per_face[4], 1), minval=self.y_min, maxval=self.y_max)
                z = jax.random.uniform(keys[3], (points_per_face[4], 1), minval=self.z_min, maxval=self.z_max)
                x = jnp.full((points_per_face[4], 1), self.x_min)
                all_points.append(jnp.hstack([x, y, z]))
            
            # Right (x=x_max)
            if points_per_face[5] > 0:
                y = jax.random.uniform(keys[4], (points_per_face[5], 1), minval=self.y_min, maxval=self.y_max)
                z = jax.random.uniform(keys[5], (points_per_face[5], 1), minval=self.z_min, maxval=self.z_max)
                x = jnp.full((points_per_face[5], 1), self.x_max)
                all_points.append(jnp.hstack([x, y, z]))
            
            return jnp.vstack(all_points)
        
        raise ValueError(f'Unknown method: {method}')

    def generate_collocation(self, n_interior=500, n_boundary=100,
                             method_interior='random', method_boundary='random',
                             rng=None):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])


class Sphere(Geometry):
    """
    Трехмерная сфера с центром в (cx, cy, cz) и радиусом R
    """

    def __init__(self, cx=0.0, cy=0.0, cz=0.0, radius=1.0):
        super().__init__(dim=3)
        self.cx = float(cx)
        self.cy = float(cy)
        self.cz = float(cz)
        self.radius = float(radius)

    def sample_interior(self, n_points, method='random', rng=None):
        """Генерирует точки внутри сферы"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == 'random':
            keys = jax.random.split(rng, 3)
            # Равномерное распределение в сфере через сферические координаты
            r = self.radius * jnp.cbrt(jax.random.uniform(keys[0], (n_points, 1)))
            theta = 2 * jnp.pi * jax.random.uniform(keys[1], (n_points, 1))
            phi = jnp.arccos(2 * jax.random.uniform(keys[2], (n_points, 1)) - 1)
            
            x = self.cx + r * jnp.sin(phi) * jnp.cos(theta)
            y = self.cy + r * jnp.sin(phi) * jnp.sin(theta)
            z = self.cz + r * jnp.cos(phi)
            return jnp.hstack([x, y, z])
        
        raise ValueError(f'Unknown method: {method}')

    def sample_boundary(self, n_points=100, method='random', rng=None):
        """Генерирует точки на поверхности сферы"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == 'random':
            keys = jax.random.split(rng, 2)
            theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n_points, 1))
            phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n_points, 1)) - 1)
            
            x = self.cx + self.radius * jnp.sin(phi) * jnp.cos(theta)
            y = self.cy + self.radius * jnp.sin(phi) * jnp.sin(theta)
            z = self.cz + self.radius * jnp.cos(phi)
            return jnp.hstack([x, y, z])
        
        raise ValueError(f'Unknown method: {method}')

    def generate_collocation(self, n_interior=500, n_boundary=100,
                             method_interior='random', method_boundary='random',
                             rng=None):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])