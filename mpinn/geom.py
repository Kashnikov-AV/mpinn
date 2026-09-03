from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def plot_domain(geometry, interior_points, boundary_points=None, interface_points=None, title="Domain Visualization", **kwargs):
    """
    Визуализирует домен с точками коллокации.

    Args:
        geometry: объект GeometryBase для определения размерности
        interior_points: внутренние точки (N, dim)
        boundary_points: граничные точки (M, dim), опционально
        interface_points: точки на интерфейсах доменов (K, dim), опционально
        title: заголовок графика
        **kwargs: дополнительные аргументы для специфичных типов геометрии (например, show_normals, normal_scale)
    """
    if geometry.dim == 1:
        _plot_1d(interior_points, boundary_points, interface_points, title)
    elif geometry.dim == 2:
        _plot_2d(interior_points, boundary_points, interface_points, title, **kwargs)
    elif geometry.dim == 3:
        _plot_3d(interior_points, boundary_points, interface_points, title, **kwargs)
    else:
        raise ValueError(f"Visualization not supported for {geometry.dim}D")


def _plot_1d(interior, boundary=None, interface=None, title="Domain Visualization"):
    """Отрисовка 1D домена"""
    plt.figure(figsize=(10, 2))

    # Внутренние точки - синие
    if interior is not None and len(interior) > 0:
        plt.scatter(
            interior[:, 0],
            jnp.zeros_like(interior[:, 0]),
            c="blue",
            s=30,
            alpha=0.5,
            label="Внутренние точки",
        )

    # Граничные точки - зеленые
    if boundary is not None and len(boundary) > 0:
        if isinstance(boundary, tuple):
            boundary_pts, _ = boundary
        else:
            boundary_pts = boundary
        plt.scatter(
            boundary_pts[:, 0],
            jnp.zeros_like(boundary_pts[:, 0]),
            c="green",
            s=50,
            alpha=0.7,
            label="Граница",
        )

    # Точки на интерфейсах - красные
    if interface is not None and len(interface) > 0:
        plt.scatter(
            interface[:, 0],
            jnp.zeros_like(interface[:, 0]),
            c="red",
            s=50,
            alpha=0.7,
            label="Интерфейс",
        )

    plt.xlabel("x")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


def _plot_2d(interior, boundary=None, interface=None, title="Domain Visualization", show_normals=False, scale=0.1):
    """Отрисовка 2D домена"""
    plt.figure(figsize=(8, 6))

    # Внутренние точки - синие
    if interior is not None and len(interior) > 0:
        plt.scatter(
            interior[:, 0],
            interior[:, 1],
            c="blue",
            s=10,
            alpha=0.5,
            label="Внутренние точки",
        )

    # Граничные точки - зеленые
    if boundary is not None and len(boundary) > 0:
        if isinstance(boundary, tuple):
            boundary_pts, normals = boundary
            plt.scatter(
                boundary_pts[:, 0],
                boundary_pts[:, 1],
                c="green",
                s=20,
                alpha=0.7,
                label="Граница",
            )

            if show_normals:
                for i in range(min(50, len(boundary_pts))):
                    pt = boundary_pts[i]
                    n = normals[i]
                    plt.arrow(
                        pt[0],
                        pt[1],
                        scale * n[0],
                        scale * n[1],
                        head_width=0.02,
                        head_length=0.03,
                        fc="red",
                        ec="red",
                    )
        else:
            plt.scatter(
                boundary[:, 0],
                boundary[:, 1],
                c="green",
                s=20,
                alpha=0.7,
                label="Граница",
            )

    # Точки на интерфейсах - красные
    if interface is not None and len(interface) > 0:
        plt.scatter(
            interface[:, 0],
            interface[:, 1],
            c="red",
            s=20,
            alpha=0.7,
            label="Интерфейс",
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def _plot_3d(interior, boundary=None, interface=None, title="Domain Visualization", show_normals=False, scale=0.1):
    """Отрисовка 3D домена"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Внутренние точки - синие
    if interior is not None and len(interior) > 0:
        ax.scatter(
            interior[:, 0],
            interior[:, 1],
            interior[:, 2],
            c="blue",
            s=10,
            alpha=0.5,
            label="Внутренние точки",
        )

    # Граничные точки - зеленые
    if boundary is not None and len(boundary) > 0:
        if isinstance(boundary, tuple):
            boundary_pts, _ = boundary
        else:
            boundary_pts = boundary
        ax.scatter(
            boundary_pts[:, 0],
            boundary_pts[:, 1],
            boundary_pts[:, 2],
            c="green",
            s=20,
            alpha=0.7,
            label="Граница",
        )

    # Точки на интерфейсах - красные
    if interface is not None and len(interface) > 0:
        ax.scatter(
            interface[:, 0],
            interface[:, 1],
            interface[:, 2],
            c="red",
            s=20,
            alpha=0.7,
            label="Интерфейс",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.show()

class GeometryBase(ABC):
    def __init__(self, dim):
        self.dim = dim

    @abstractmethod
    def sample_interior(self, n_points, method="random", rng=None):
        pass

    @abstractmethod
    def sample_boundary(self, n_points=None, method="random", rng=None):
        """
        Генерирует точки на границе и соответствующие нормали.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Кортеж (points, normals), где:
            - points: массив координат формы (N, D)
            - normals: массив единичных нормалей формы (N, D)
        """
        pass


class Interval(GeometryBase):
    def __init__(self, x0, x1):
        super().__init__(dim=1)
        self.x0 = float(x0)
        self.x1 = float(x1)

    def sample_interior(self, n_points, method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            return jax.random.uniform(
                rng, shape=(n_points, 1), minval=self.x0, maxval=self.x1
            )
        if method == "uniform":
            return jnp.linspace(self.x0, self.x1, n_points).reshape(-1, 1)
        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self, n_points=None, method="random", rng=None):
        """
        Генерирует точки на границе интервала и соответствующие нормали.

        В 1D случае нормаль - это скаляр: -1 для левой границы, +1 для правой.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Кортеж (points, normals), где:
            - points: массив координат формы (2, 1)
            - normals: массив нормалей формы (2, 1), [-1, 0], [1, 0]
        """
        points = jnp.array([[self.x0], [self.x1]])
        normals = jnp.array([[-1.0], [1.0]])
        return points, normals

    def generate_collocation(self, n_interior=100, method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)

        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method, keys[0])
        boundary, _ = self.sample_boundary()

        return jnp.vstack([interior, boundary])

class Box2D(GeometryBase):
    """
    Двумерный прямоугольник x_min, x_max × y_min, y_max.
    """

    def __init__(self, x_min, x_max, y_min, y_max):
        super().__init__(dim=2)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def sample_interior(self, n_points, method="random", rng=None):
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 2)
            x = jax.random.uniform(
                keys[0], (n_points, 1), minval=self.x_min, maxval=self.x_max
            )
            y = jax.random.uniform(
                keys[1], (n_points, 1), minval=self.y_min, maxval=self.y_max
            )
            return jnp.hstack([x, y])
        if method == "uniform":
            n_per_side = int(jnp.sqrt(n_points))
            x = jnp.linspace(self.x_min, self.x_max, n_per_side)
            y = jnp.linspace(self.y_min, self.y_max, n_per_side)
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
            return points[:n_points]
        raise ValueError(f"Неизвестный метод: {method}")

    def sample_boundary(self, n_points, method="random", rng=None, return_edges=False):
        """
        Генерирует точки на границе. Точки распределяются по сторонам пропорционально длинам.
        n_points должно быть > 0.
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        def _points_on_segment(start, end, n, rng_sub):
            if n <= 0:
                return jnp.empty((0, 2))
            if method == "random":
                t = jax.random.uniform(rng_sub, (n, 1), minval=0.0, maxval=1.0)
            else:
                t = jnp.linspace(0, 1, n).reshape(-1, 1)
            return start + t * (end - start)

        sides = {
            'left':   (jnp.array([self.x_min, self.y_min]), jnp.array([self.x_min, self.y_max]), jnp.array([-1.0, 0.0])),
            'right':  (jnp.array([self.x_max, self.y_min]), jnp.array([self.x_max, self.y_max]), jnp.array([ 1.0, 0.0])),
            'bottom': (jnp.array([self.x_min, self.y_min]), jnp.array([self.x_max, self.y_min]), jnp.array([ 0.0,-1.0])),
            'top':    (jnp.array([self.x_min, self.y_max]), jnp.array([self.x_max, self.y_max]), jnp.array([ 0.0, 1.0])),
        }

        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        perim = 2 * (dx + dy)
        lengths = jnp.array([dy, dy, dx, dx])
        fractions = lengths / perim
        n_side = jnp.round(fractions * n_points).astype(int)
        diff = n_points - jnp.sum(n_side)
        if diff != 0:
            idx = jnp.argmax(lengths)
            n_side = n_side.at[idx].add(diff)

        keys = jax.random.split(rng, 4) if method == "random" else [None] * 4
        result = {}
        side_names = ['left', 'right', 'bottom', 'top']
        for name, (start, end, normal), n, k in zip(side_names, sides.values(), n_side, keys):
            pts = _points_on_segment(start, end, int(n), k)
            norms = jnp.tile(normal.reshape(1, 2), (int(n), 1)) if n > 0 else jnp.empty((0, 2))
            result[name] = (pts, norms)

        if return_edges:
            return result
        else:
            all_pts = jnp.vstack([v[0] for v in result.values()])
            all_norms = jnp.vstack([v[1] for v in result.values()])
            return all_pts, all_norms

    def generate_collocation(
        self,
        n_interior,
        n_boundary,
        method_interior="random",
        method_boundary="random",
        rng=None,
    ):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary, _ = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])


class Box3D(GeometryBase):
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

    def sample_interior(self, n_points, method="random", rng=None):
        """Генерирует точки внутри параллелепипеда"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 3)
            x = jax.random.uniform(
                keys[0], (n_points, 1), minval=self.x_min, maxval=self.x_max
            )
            y = jax.random.uniform(
                keys[1], (n_points, 1), minval=self.y_min, maxval=self.y_max
            )
            z = jax.random.uniform(
                keys[2], (n_points, 1), minval=self.z_min, maxval=self.z_max
            )
            return jnp.hstack([x, y, z])

        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self, n_points, method="random", rng=None, return_faces=False):
        """Генерирует точки на границе параллелепипеда и нормали."""
        if n_points < 6:
            raise ValueError("n_points должен быть >= 6")

        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method != "random":
            raise ValueError(f"Неизвестный метод: {method}")

        dx = self.x_max - self.x_min
        dy = self.y_max - self.y_min
        dz = self.z_max - self.z_min

        # Площади граней: bottom, top, front, back, left, right
        areas = jnp.array([dx*dy, dx*dy, dx*dz, dx*dz, dy*dz, dy*dz])
        
        # Каждая грань получает минимум 1 точку, остаток пропорционально площадям
        n_face = jnp.round((areas / jnp.sum(areas)) * (n_points - 6)).astype(int) + 1
        diff = n_points - jnp.sum(n_face)
        if diff != 0:
            idx = jnp.argmax(areas)
            n_face = n_face.at[idx].add(diff)

        keys = jax.random.split(rng, 6)
        face_names = ['bottom', 'top', 'front', 'back', 'left', 'right']
        all_points = []
        all_normals = []
        faces_data = {name: ([], []) for name in face_names}

        # bottom (z = z_min)
        n = int(n_face[0])
        x = jax.random.uniform(keys[0], (n, 1), minval=self.x_min, maxval=self.x_max)
        y = jax.random.uniform(keys[1], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jnp.full((n, 1), self.z_min)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, 0.0, -1.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['bottom'][0].append(pts)
        faces_data['bottom'][1].append(norms)

        # top (z = z_max)
        n = int(n_face[1])
        x = jax.random.uniform(keys[2], (n, 1), minval=self.x_min, maxval=self.x_max)
        y = jax.random.uniform(keys[3], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jnp.full((n, 1), self.z_max)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, 0.0, 1.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['top'][0].append(pts)
        faces_data['top'][1].append(norms)

        # front (y = y_min)
        n = int(n_face[2])
        x = jax.random.uniform(keys[4], (n, 1), minval=self.x_min, maxval=self.x_max)
        z = jax.random.uniform(keys[5], (n, 1), minval=self.z_min, maxval=self.z_max)
        y = jnp.full((n, 1), self.y_min)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, -1.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['front'][0].append(pts)
        faces_data['front'][1].append(norms)

        # back (y = y_max)
        n = int(n_face[3])
        x = jax.random.uniform(keys[0], (n, 1), minval=self.x_min, maxval=self.x_max)
        z = jax.random.uniform(keys[1], (n, 1), minval=self.z_min, maxval=self.z_max)
        y = jnp.full((n, 1), self.y_max)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([0.0, 1.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['back'][0].append(pts)
        faces_data['back'][1].append(norms)

        # left (x = x_min)
        n = int(n_face[4])
        y = jax.random.uniform(keys[2], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jax.random.uniform(keys[3], (n, 1), minval=self.z_min, maxval=self.z_max)
        x = jnp.full((n, 1), self.x_min)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([-1.0, 0.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['left'][0].append(pts)
        faces_data['left'][1].append(norms)

        # right (x = x_max)
        n = int(n_face[5])
        y = jax.random.uniform(keys[4], (n, 1), minval=self.y_min, maxval=self.y_max)
        z = jax.random.uniform(keys[5], (n, 1), minval=self.z_min, maxval=self.z_max)
        x = jnp.full((n, 1), self.x_max)
        pts = jnp.hstack([x, y, z])
        norms = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (n, 1))
        all_points.append(pts)
        all_normals.append(norms)
        faces_data['right'][0].append(pts)
        faces_data['right'][1].append(norms)

        if return_faces:
            return {name: (jnp.vstack(faces_data[name][0]), jnp.vstack(faces_data[name][1])) 
                    for name in face_names}
        else:
            return jnp.vstack(all_points), jnp.vstack(all_normals)

    def generate_collocation(
        self,
        n_interior,
        n_boundary,
        method_interior="random",
        method_boundary="random",
        rng=None,
    ):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary, _ = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])

class Sphere(GeometryBase):
    """
    Трехмерная сфера с центром в (cx, cy, cz) и радиусом R
    """

    def __init__(self, cx=0.0, cy=0.0, cz=0.0, radius=1.0):
        super().__init__(dim=3)
        self.cx = float(cx)
        self.cy = float(cy)
        self.cz = float(cz)
        self.radius = float(radius)

    def sample_interior(self, n_points, method="random", rng=None):
        """Генерирует точки внутри сферы"""
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 3)
            # Равномерное распределение в сфере через сферические координаты
            r = self.radius * jnp.cbrt(jax.random.uniform(keys[0], (n_points, 1)))
            theta = 2 * jnp.pi * jax.random.uniform(keys[1], (n_points, 1))
            phi = jnp.arccos(2 * jax.random.uniform(keys[2], (n_points, 1)) - 1)

            x = self.cx + r * jnp.sin(phi) * jnp.cos(theta)
            y = self.cy + r * jnp.sin(phi) * jnp.sin(theta)
            z = self.cz + r * jnp.cos(phi)
            return jnp.hstack([x, y, z])

        raise ValueError(f"Unknown method: {method}")

    def sample_boundary(self, n_points=100, method="random", rng=None):
        """
        Генерирует точки на поверхности сферы и нормали.

        Returns:
            tuple[jax.Array, jax.Array]:
                - points: массив формы (n_points, 3)
                - normals: массив единичных нормалей формы (n_points, 3)
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)

        if method == "random":
            keys = jax.random.split(rng, 2)
            theta = 2 * jnp.pi * jax.random.uniform(keys[0], (n_points,))
            phi = jnp.arccos(2 * jax.random.uniform(keys[1], (n_points,)) - 1)

            sin_phi = jnp.sin(phi)
            cos_phi = jnp.cos(phi)
            sin_theta = jnp.sin(theta)
            cos_theta = jnp.cos(theta)

            x = self.cx + self.radius * sin_phi[:, None] * cos_theta[:, None]
            y = self.cy + self.radius * sin_phi[:, None] * sin_theta[:, None]
            z = self.cz + self.radius * cos_phi[:, None]
            
            points = jnp.hstack([x, y, z])
            
            # Нормали направлены радиально наружу
            normals = jnp.hstack([
                sin_phi[:, None] * cos_theta[:, None],
                sin_phi[:, None] * sin_theta[:, None],
                cos_phi[:, None]
            ])
            
            return points, normals

        raise ValueError(f"Unknown method: {method}")

    def generate_collocation(
        self,
        n_interior,
        n_boundary,
        method_interior="random",
        method_boundary="random",
        rng=None,
    ):
        """Генерирует внутренние и граничные точки"""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 2)
        interior = self.sample_interior(n_interior, method_interior, keys[0])
        boundary, _ = self.sample_boundary(n_boundary, method_boundary, keys[1])
        return jnp.vstack([interior, boundary])

class Annulus2D(GeometryBase):
    """
    2D кольцо (полый круг) с внутренним и внешним радиусами.

    Параметры:
        center: центр (x, y)
        R_outer: внешний радиус (> 0)
        R_inner: внутренний радиус (> 0)
    """

    def __init__(
        self, center: tuple[float, float], R_outer: float, R_inner: float = 0.0
    ):
        if R_outer <= 0:
            raise ValueError("R_outer должен быть > 0")
        if R_inner <= 0:
            raise ValueError("R_inner должен быть > 0")
        if R_inner >= R_outer:
            raise ValueError("R_inner должен быть < R_outer")

        super().__init__(dim=2)
        self.center = jnp.array(center, dtype=jnp.float64)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)
        self.is_hollow = R_inner > 0

    def sample_interior(
        self, n_points: int, rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        r_max = self.R_outer
        r_min = self.R_inner

        u = jax.random.uniform(rng, (n_points,))
        theta = jax.random.uniform(rng, (n_points,), minval=0, maxval=2 * jnp.pi)

        if self.is_hollow:
            r = jnp.sqrt(r_min**2 + u * (r_max**2 - r_min**2))
        else:
            r = jnp.sqrt(u) * r_max

        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)

        return jnp.column_stack([x, y])

    def sample_boundary(
        self, n_points: int, rng: jax.Array | None = None, return_edges: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray] | dict:
        """
        Генерирует точки на границе кольца и нормали.

        Args:
            n_points: общее количество точек (распределяются поровну между границами)
            rng: ключ PRNG
            return_edges: если True, возвращает словарь с ключами 'outer' и 'inner'

        Returns:
            Если return_edges=False: (points, normals) – объединённые массивы
            Если return_edges=True: словарь с ключами 'outer' и 'inner'
        """
        if n_points <= 0:
            raise ValueError("Количество точек должно быть положительным")

        if rng is None:
            rng = jax.random.PRNGKey(1)

        keys = jax.random.split(rng, 2)

        # Если кольцо не полое - только внешняя граница
        if not self.is_hollow:
            theta = jax.random.uniform(
                keys[0], (n_points,), minval=0, maxval=2 * jnp.pi
            )
            x = self.center[0] + self.R_outer * jnp.cos(theta)
            y = self.center[1] + self.R_outer * jnp.sin(theta)
            points = jnp.column_stack([x, y])
            normals = jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
            
            if return_edges:
                return {'outer': (points, normals)}
            else:
                return points, normals

        # Полое кольцо: распределяем точки поровну
        n_outer = n_points // 2
        n_inner = n_points - n_outer  # остаток отдаём внутренней границе

        # Внешняя граница
        theta_out = jax.random.uniform(
            keys[0], (n_outer,), minval=0, maxval=2 * jnp.pi
        )
        x_out = self.center[0] + self.R_outer * jnp.cos(theta_out)
        y_out = self.center[1] + self.R_outer * jnp.sin(theta_out)
        points_outer = jnp.column_stack([x_out, y_out])
        normals_outer = jnp.column_stack([jnp.cos(theta_out), jnp.sin(theta_out)])

        # Внутренняя граница
        theta_in = jax.random.uniform(
            keys[1], (n_inner,), minval=0, maxval=2 * jnp.pi
        )
        x_in = self.center[0] + self.R_inner * jnp.cos(theta_in)
        y_in = self.center[1] + self.R_inner * jnp.sin(theta_in)
        points_inner = jnp.column_stack([x_in, y_in])
        # Нормаль направлена внутрь полости (наружу из материала)
        normals_inner = jnp.column_stack([-jnp.cos(theta_in), -jnp.sin(theta_in)])

        if return_edges:
            return {
                'outer': (points_outer, normals_outer),
                'inner': (points_inner, normals_inner)
            }
        else:
            points = jnp.vstack([points_outer, points_inner])
            normals = jnp.vstack([normals_outer, normals_inner])
            return points, normals

    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        min_corner = self.center - self.R_outer
        max_corner = self.center + self.R_outer
        return min_corner, max_corner
    

class HollowCylinder3D(GeometryBase):
    """
    3D полый цилиндр (труба) с внутренним и внешним радиусами.

    Параметры:
        center_base: центр основания (x, y, z)
        radius_outer: внешний радиус (> 0)
        radius_inner: внутренний радиус (> 0, строго полый)
        height: высота цилиндра
        axis: ось направления (по умолчанию Z)
    """

    def __init__(
        self,
        center_base: tuple[float, float, float],
        radius_outer: float,
        radius_inner: float,
        height: float,
        axis: tuple[float, float, float] = (0, 0, 1),
    ):
        if radius_outer <= 0:
            raise ValueError("radius_outer должен быть > 0")
        if radius_inner <= 0:
            raise ValueError("radius_inner должен быть > 0 (строго полый)")
        if radius_inner >= radius_outer:
            raise ValueError("radius_inner должен быть < radius_outer")
        if height <= 0:
            raise ValueError("height должен быть > 0")

        super().__init__(dim=3)
        self.center_base = jnp.array(center_base, dtype=jnp.float64)
        self.radius_outer = float(radius_outer)
        self.radius_inner = float(radius_inner)
        self.height = float(height)
        self.axis = jnp.array(axis, dtype=jnp.float64)
        self.axis = self.axis / jnp.linalg.norm(self.axis)  # Нормализация

    def get_boundary_tags(self) -> dict[str, str]:
        return {
            "outer_lateral": "exterior_boundary",
            "inner_lateral": "interface",
            "bottom": "exterior_boundary",
            "top": "exterior_boundary",
        }

    def _cylindrical_to_cartesian(
        self, r: jnp.ndarray, theta: jnp.ndarray, z: jnp.ndarray
    ) -> jnp.ndarray:
        """Преобразование цилиндрических координат в декартовы."""
        # Локальная система координат
        ez = self.axis

        # Базисные векторы перпендикулярные оси
        if jnp.abs(ez[2]) < 0.9:
            er_ref = jnp.cross(ez, jnp.array([0.0, 0.0, 1.0]))
        else:
            er_ref = jnp.cross(ez, jnp.array([1.0, 0.0, 0.0]))
        er_ref = er_ref / jnp.linalg.norm(er_ref)
        etheta_ref = jnp.cross(ez, er_ref)

        # Векторизованное преобразование
        x_local = (
            (r * jnp.cos(theta))[:, None] * er_ref
            + (r * jnp.sin(theta))[:, None] * etheta_ref
            + z[:, None] * ez
        )

        return self.center_base + x_local

    def sample_interior(
        self, n_points: int, rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        keys = jax.random.split(rng, 3)

        # Равномерное распределение по объему
        u = jax.random.uniform(keys[0], (n_points,))
        theta = jax.random.uniform(keys[1], (n_points,), minval=0, maxval=2 * jnp.pi)
        z = jax.random.uniform(keys[2], (n_points,), minval=0, maxval=self.height)

        r = jnp.sqrt(
            self.radius_inner**2 + u * (self.radius_outer**2 - self.radius_inner**2)
        )

        return self._cylindrical_to_cartesian(r, theta, z)

    def sample_boundary(
        self, n_points: int, rng: jax.Array | None = None, tags: list | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(1)

        keys = jax.random.split(rng, 4)
        points_list = []
        normals_list = []

        # Распределение точек по поверхностям
        n_per_surface = n_points // 4

        # 1. Внешняя боковая поверхность
        if tags is None or "outer_lateral" in tags:
            theta = jax.random.uniform(
                keys[0], (n_per_surface,), minval=0, maxval=2 * jnp.pi
            )
            z = jax.random.uniform(
                keys[1], (n_per_surface,), minval=0, maxval=self.height
            )
            r = jnp.full(n_per_surface, self.radius_outer)

            pts = self._cylindrical_to_cartesian(r, theta, z)
            points_list.append(pts)

            # Нормаль радиально наружу
            jnp.array([1.0, 0.0, 0.0])
            ez = self.axis
            if jnp.abs(ez[2]) < 0.9:
                er_ref = jnp.cross(ez, jnp.array([0.0, 0.0, 1.0]))
            else:
                er_ref = jnp.cross(ez, jnp.array([1.0, 0.0, 0.0]))
            er_ref = er_ref / jnp.linalg.norm(er_ref)
            etheta_ref = jnp.cross(ez, er_ref)

            norm = (
                jnp.cos(theta)[:, None] * er_ref + jnp.sin(theta)[:, None] * etheta_ref
            )
            normals_list.append(norm)

        # 2. Внутренняя боковая поверхность
        if tags is None or "inner_lateral" in tags:
            theta = jax.random.uniform(
                keys[2], (n_per_surface,), minval=0, maxval=2 * jnp.pi
            )
            z = jax.random.uniform(
                keys[3], (n_per_surface,), minval=0, maxval=self.height
            )
            r = jnp.full(n_per_surface, self.radius_inner)

            pts = self._cylindrical_to_cartesian(r, theta, z)
            points_list.append(pts)

            # Нормаль радиально внутрь (наружу из материала)
            norm = -(
                jnp.cos(theta)[:, None] * er_ref + jnp.sin(theta)[:, None] * etheta_ref
            )
            normals_list.append(norm)

        # 3. Нижнее основание
        # ... (аналогично для top и bottom)

        return jnp.vstack(points_list), jnp.vstack(normals_list)

    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        # Проверка принадлежности точке к полому цилиндру
        vec = points - self.center_base
        z_coord = jnp.dot(vec, self.axis)
        r_vec = vec - z_coord[:, None] * self.axis
        r = jnp.linalg.norm(r_vec, axis=1)

        return (
            (r <= self.radius_outer)
            & (r >= self.radius_inner)
            & (z_coord >= 0)
            & (z_coord <= self.height)
        )

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        R = self.radius_outer
        H = self.height
        # Приблизительные границы (bounding box)
        min_corner = self.center_base - jnp.array([R, R, 0])
        max_corner = self.center_base + jnp.array([R, R, H])
        return min_corner, max_corner

class HollowSphere3D(GeometryBase):
    """
    3D полая сфера с внутренним и внешним радиусами.

    Параметры:
        center: центр сферы (x, y, z)
        R_outer: внешний радиус (> 0)
        R_inner: внутренний радиус (> 0, строго полый)
    """

    def __init__(
        self, center: tuple[float, float, float], R_outer: float, R_inner: float
    ):
        if R_outer <= 0:
            raise ValueError("R_outer должен быть > 0")
        if R_inner <= 0:
            raise ValueError("R_inner должен быть > 0 (строго полый)")
        if R_inner >= R_outer:
            raise ValueError("R_inner должен быть < R_outer")

        super().__init__(dim=3)
        self.center = jnp.array(center, dtype=jnp.float64)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)

    def get_boundary_tags(self) -> dict[str, str]:
        return {"outer": "exterior_boundary", "inner": "interface"}

    def sample_interior(
        self, n_points: int, rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)

        keys = jax.random.split(rng, 3)

        # Сферические координаты с равномерным распределением по объему
        u = jax.random.uniform(keys[0], (n_points,))
        v = jax.random.uniform(keys[1], (n_points,))
        w = jax.random.uniform(keys[2], (n_points,))

        theta = 2 * jnp.pi * u
        phi = jnp.arccos(2 * v - 1)
        r = (self.R_inner**3 + w * (self.R_outer**3 - self.R_inner**3)) ** (1 / 3)

        x = self.center[0] + r * jnp.sin(phi) * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(phi) * jnp.sin(theta)
        z = self.center[2] + r * jnp.cos(phi)

        return jnp.column_stack([x, y, z])

    def sample_boundary(
        self, n_points: int, rng: jax.Array | None = None, tags: list | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(1)

        keys = jax.random.split(rng, 2)
        points_list = []
        normals_list = []

        n_per_surface = n_points // 2

        # Внешняя сфера
        if tags is None or "outer" in tags:
            u = jax.random.uniform(keys[0], (n_per_surface,))
            v = jax.random.uniform(keys[1], (n_per_surface,))

            theta = 2 * jnp.pi * u
            phi = jnp.arccos(2 * v - 1)

            x = self.center[0] + self.R_outer * jnp.sin(phi) * jnp.cos(theta)
            y = self.center[1] + self.R_outer * jnp.sin(phi) * jnp.sin(theta)
            z = self.center[2] + self.R_outer * jnp.cos(phi)

            points_list.append(jnp.column_stack([x, y, z]))
            normals_list.append(
                jnp.column_stack(
                    [
                        jnp.sin(phi) * jnp.cos(theta),
                        jnp.sin(phi) * jnp.sin(theta),
                        jnp.cos(phi),
                    ]
                )
            )

        # Внутренняя сфера
        if tags is None or "inner" in tags:
            u = jax.random.uniform(keys[0], (n_per_surface,))
            v = jax.random.uniform(keys[1], (n_per_surface,))

            theta = 2 * jnp.pi * u
            phi = jnp.arccos(2 * v - 1)

            x = self.center[0] + self.R_inner * jnp.sin(phi) * jnp.cos(theta)
            y = self.center[1] + self.R_inner * jnp.sin(phi) * jnp.sin(theta)
            z = self.center[2] + self.R_inner * jnp.cos(phi)

            points_list.append(jnp.column_stack([x, y, z]))
            # Нормаль направлена внутрь (наружу из материала)
            normals_list.append(
                -jnp.column_stack(
                    [
                        jnp.sin(phi) * jnp.cos(theta),
                        jnp.sin(phi) * jnp.sin(theta),
                        jnp.cos(phi),
                    ]
                )
            )

        return jnp.vstack(points_list), jnp.vstack(normals_list)

    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)

    @property
    def bounds(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        min_corner = self.center - self.R_outer
        max_corner = self.center + self.R_outer
        return min_corner, max_corner
    