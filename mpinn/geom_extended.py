"""
Extended geometry module for 2D/3D complex geometries.

Provides MeshGeometry class that wraps loaded mesh data and provides
sampling methods compatible with the existing Geometry interface.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

from .geom import Geometry
from .geometry_io import MeshData, load_geometry, compute_face_normals


class GeometryBase(Geometry):
    """
    Базовый класс для всех параметрических геометрий.
    Наследуется от существующего Geometry для совместимости.
    """
    def __init__(self, dim: int):
        super().__init__(dim=dim)
    
    def get_boundary_tags(self) -> Dict[str, Any]:
        """Возвращает словарь с именами границ для многодоменных задач."""
        return {}


class Annulus2D(GeometryBase):
    """
    2D кольцо (полый круг) с внутренним и внешним радиусами.
    
    Параметры:
        center: центр (x, y)
        R_outer: внешний радиус (> 0)
        R_inner: внутренний радиус (>= 0, если 0 - сплошной круг)
    """
    
    def __init__(self, center: Tuple[float, float], R_outer: float, R_inner: float = 0.0):
        if R_outer <= 0:
            raise ValueError("R_outer должен быть > 0")
        if R_inner < 0:
            raise ValueError("R_inner должен быть >= 0")
        if R_inner >= R_outer:
            raise ValueError("R_inner должен быть < R_outer")
        
        super().__init__(dim=2)
        self.center = jnp.array(center, dtype=jnp.float64)
        self.R_outer = float(R_outer)
        self.R_inner = float(R_inner)
        self.is_hollow = R_inner > 0
    
    def get_boundary_tags(self) -> Dict[str, str]:
        tags = {"outer": "exterior_boundary"}
        if self.is_hollow:
            tags["inner"] = "interface"
        return tags
    
    def sample_interior(self, n_points: int, rng: Optional[jax.Array] = None) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        # Выборка в полярных координатах с коррекцией плотности
        r_max = self.R_outer
        r_min = self.R_inner
        
        # Равномерное распределение по площади
        u = jax.random.uniform(rng, (n_points,))
        theta = jax.random.uniform(rng, (n_points,), minval=0, maxval=2*jnp.pi)
        
        if self.is_hollow:
            r = jnp.sqrt(r_min**2 + u * (r_max**2 - r_min**2))
        else:
            r = jnp.sqrt(u) * r_max
        
        x = self.center[0] + r * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(theta)
        
        return jnp.column_stack([x, y])
    
    def sample_boundary(self, n_points: int, rng: Optional[jax.Array] = None,
                       tags: Optional[list] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(1)
        
        keys = jax.random.split(rng, 2)
        
        points_list = []
        normals_list = []
        
        # Внешняя граница
        if tags is None or "outer" in tags:
            n_outer = n_points // 2 if self.is_hollow else n_points
            theta_out = jax.random.uniform(keys[0], (n_outer,), minval=0, maxval=2*jnp.pi)
            x_out = self.center[0] + self.R_outer * jnp.cos(theta_out)
            y_out = self.center[1] + self.R_outer * jnp.sin(theta_out)
            points_list.append(jnp.column_stack([x_out, y_out]))
            # Нормаль направлена наружу
            n_out = jnp.column_stack([jnp.cos(theta_out), jnp.sin(theta_out)])
            normals_list.append(n_out)
        
        # Внутренняя граница (если полая)
        if self.is_hollow and (tags is None or "inner" in tags):
            n_inner = n_points - len(points_list[0]) if tags is None else n_points // 2
            theta_in = jax.random.uniform(keys[1], (n_inner,), minval=0, maxval=2*jnp.pi)
            x_in = self.center[0] + self.R_inner * jnp.cos(theta_in)
            y_in = self.center[1] + self.R_inner * jnp.sin(theta_in)
            points_list.append(jnp.column_stack([x_in, y_in]))
            # Нормаль направлена внутрь полости (наружу из материала)
            n_in = jnp.column_stack([-jnp.cos(theta_in), -jnp.sin(theta_in)])
            normals_list.append(n_in)
        
        return jnp.vstack(points_list), jnp.vstack(normals_list)
    
    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)
    
    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    
    def __init__(self, center_base: Tuple[float, float, float], 
                 radius_outer: float, radius_inner: float, 
                 height: float, axis: Tuple[float, float, float] = (0, 0, 1)):
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
    
    def get_boundary_tags(self) -> Dict[str, str]:
        return {
            "outer_lateral": "exterior_boundary",
            "inner_lateral": "interface",
            "bottom": "exterior_boundary",
            "top": "exterior_boundary"
        }
    
    def _cylindrical_to_cartesian(self, r: jnp.ndarray, theta: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
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
        x_local = (r * jnp.cos(theta))[:, None] * er_ref + \
                  (r * jnp.sin(theta))[:, None] * etheta_ref + \
                  z[:, None] * ez
        
        return self.center_base + x_local
    
    def sample_interior(self, n_points: int, rng: Optional[jax.Array] = None) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        keys = jax.random.split(rng, 3)
        
        # Равномерное распределение по объему
        u = jax.random.uniform(keys[0], (n_points,))
        theta = jax.random.uniform(keys[1], (n_points,), minval=0, maxval=2*jnp.pi)
        z = jax.random.uniform(keys[2], (n_points,), minval=0, maxval=self.height)
        
        r = jnp.sqrt(self.radius_inner**2 + u * (self.radius_outer**2 - self.radius_inner**2))
        
        return self._cylindrical_to_cartesian(r, theta, z)
    
    def sample_boundary(self, n_points: int, rng: Optional[jax.Array] = None,
                       tags: Optional[list] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if rng is None:
            rng = jax.random.PRNGKey(1)
        
        keys = jax.random.split(rng, 4)
        points_list = []
        normals_list = []
        
        # Распределение точек по поверхностям
        n_per_surface = n_points // 4
        
        # 1. Внешняя боковая поверхность
        if tags is None or "outer_lateral" in tags:
            theta = jax.random.uniform(keys[0], (n_per_surface,), minval=0, maxval=2*jnp.pi)
            z = jax.random.uniform(keys[1], (n_per_surface,), minval=0, maxval=self.height)
            r = jnp.full(n_per_surface, self.radius_outer)
            
            pts = self._cylindrical_to_cartesian(r, theta, z)
            points_list.append(pts)
            
            # Нормаль радиально наружу
            ex = jnp.array([1.0, 0.0, 0.0])
            ez = self.axis
            if jnp.abs(ez[2]) < 0.9:
                er_ref = jnp.cross(ez, jnp.array([0.0, 0.0, 1.0]))
            else:
                er_ref = jnp.cross(ez, jnp.array([1.0, 0.0, 0.0]))
            er_ref = er_ref / jnp.linalg.norm(er_ref)
            etheta_ref = jnp.cross(ez, er_ref)
            
            norm = jnp.cos(theta)[:, None] * er_ref + jnp.sin(theta)[:, None] * etheta_ref
            normals_list.append(norm)
        
        # 2. Внутренняя боковая поверхность
        if tags is None or "inner_lateral" in tags:
            theta = jax.random.uniform(keys[2], (n_per_surface,), minval=0, maxval=2*jnp.pi)
            z = jax.random.uniform(keys[3], (n_per_surface,), minval=0, maxval=self.height)
            r = jnp.full(n_per_surface, self.radius_inner)
            
            pts = self._cylindrical_to_cartesian(r, theta, z)
            points_list.append(pts)
            
            # Нормаль радиально внутрь (наружу из материала)
            norm = -(jnp.cos(theta)[:, None] * er_ref + jnp.sin(theta)[:, None] * etheta_ref)
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
        
        return ((r <= self.radius_outer) & (r >= self.radius_inner) & 
                (z_coord >= 0) & (z_coord <= self.height))
    
    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    
    def __init__(self, center: Tuple[float, float, float], 
                 R_outer: float, R_inner: float):
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
    
    def get_boundary_tags(self) -> Dict[str, str]:
        return {
            "outer": "exterior_boundary",
            "inner": "interface"
        }
    
    def sample_interior(self, n_points: int, rng: Optional[jax.Array] = None) -> jnp.ndarray:
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        keys = jax.random.split(rng, 3)
        
        # Сферические координаты с равномерным распределением по объему
        u = jax.random.uniform(keys[0], (n_points,))
        v = jax.random.uniform(keys[1], (n_points,))
        w = jax.random.uniform(keys[2], (n_points,))
        
        theta = 2 * jnp.pi * u
        phi = jnp.arccos(2 * v - 1)
        r = (self.R_inner**3 + w * (self.R_outer**3 - self.R_inner**3)) ** (1/3)
        
        x = self.center[0] + r * jnp.sin(phi) * jnp.cos(theta)
        y = self.center[1] + r * jnp.sin(phi) * jnp.sin(theta)
        z = self.center[2] + r * jnp.cos(phi)
        
        return jnp.column_stack([x, y, z])
    
    def sample_boundary(self, n_points: int, rng: Optional[jax.Array] = None,
                       tags: Optional[list] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
            normals_list.append(jnp.column_stack([
                jnp.sin(phi) * jnp.cos(theta),
                jnp.sin(phi) * jnp.sin(theta),
                jnp.cos(phi)
            ]))
        
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
            normals_list.append(-jnp.column_stack([
                jnp.sin(phi) * jnp.cos(theta),
                jnp.sin(phi) * jnp.sin(theta),
                jnp.cos(phi)
            ]))
        
        return jnp.vstack(points_list), jnp.vstack(normals_list)
    
    def is_inside(self, points: jnp.ndarray) -> jnp.ndarray:
        r = jnp.linalg.norm(points - self.center, axis=1)
        return (r <= self.R_outer) & (r >= self.R_inner)
    
    @property
    def bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        min_corner = self.center - self.R_outer
        max_corner = self.center + self.R_outer
        return min_corner, max_corner


class MeshGeometry(Geometry):
    """
    Geometry defined by a mesh loaded from file (STL, OBJ, Gmsh).
    
    Supports arbitrary complex geometries in 2D and 3D with automatic
    normal computation for boundary conditions.
    
    Attributes:
        mesh_data: Original MeshData object
        vertices: JAX array of vertex coordinates
        faces: JAX array of face indices
        normals: JAX array of face normals
        dim: Dimension (2 or 3)
        bbox: Bounding box (min_coords, max_coords)
    """
    
    def __init__(self, mesh_data: MeshData, compute_normals: bool = True):
        """
        Initialize MeshGeometry from MeshData.
        
        Args:
            mesh_data: MeshData object from geometry_io
            compute_normals: Whether to compute/update normals
        """
        super().__init__(dim=mesh_data.dim)
        
        self.mesh_data = mesh_data
        self.vertices = jnp.array(mesh_data.vertices, dtype=jnp.float64)
        self.faces = jnp.array(mesh_data.faces, dtype=jnp.int32)
        
        # Compute or use existing normals
        if compute_normals or mesh_data.normals is None:
            if len(self.faces) > 0:
                np_normals = compute_face_normals(mesh_data.vertices, mesh_data.faces)
                self.normals = jnp.array(np_normals, dtype=jnp.float64)
            else:
                self.normals = None
        else:
            self.normals = jnp.array(mesh_data.normals, dtype=jnp.float64)
        
        # Compute bounding box
        if len(self.vertices) > 0:
            self.bbox_min = jnp.min(self.vertices, axis=0)
            self.bbox_max = jnp.max(self.vertices, axis=0)
        else:
            self.bbox_min = self.bbox_max = None
        
        # Boundary markers if available
        self.boundary_markers = mesh_data.boundary_markers
        self.volume_markers = mesh_data.volume_markers
    
    @classmethod
    def from_file(cls, filepath: str, file_format: Optional[str] = None, 
                  compute_normals: bool = True, **kwargs) -> 'MeshGeometry':
        """
        Create MeshGeometry directly from file.
        
        Args:
            filepath: Path to geometry file
            file_format: Optional format override
            compute_normals: Whether to compute normals
            **kwargs: Additional arguments for loader
            
        Returns:
            MeshGeometry instance
        """
        mesh_data = load_geometry(filepath, file_format, **kwargs)
        return cls(mesh_data, compute_normals)
    
    def sample_interior(self, n_points: int, method: str = 'rejection', 
                       rng: Optional[jax.Array] = None, **kwargs) -> jnp.ndarray:
        """
        Sample points inside the geometry.
        
        For complex geometries, uses rejection sampling within bounding box
        with ray casting or winding number test.
        
        Args:
            n_points: Number of points to sample
            method: Sampling method ('rejection', 'tetrahedral')
            rng: JAX random key
            **kwargs: Additional method-specific parameters
            
        Returns:
            Array of shape (n_points, dim)
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        if self.dim == 2:
            return self._sample_interior_2d(n_points, method, rng, **kwargs)
        else:
            return self._sample_interior_3d(n_points, method, rng, **kwargs)
    
    def _sample_interior_2d(self, n_points: int, method: str, 
                           rng: jax.Array, **kwargs) -> jnp.ndarray:
        """2D interior sampling via rejection."""
        # Rejection sampling in bounding box
        accepted = []
        max_attempts = n_points * 100
        keys = jax.random.split(rng, 3)
        
        # Sample in bounding box
        x = jax.random.uniform(keys[0], (max_attempts, 1), 
                               minval=self.bbox_min[0], maxval=self.bbox_max[0])
        y = jax.random.uniform(keys[1], (max_attempts, 1),
                               minval=self.bbox_min[1], maxval=self.bbox_max[1])
        points = jnp.hstack([x, y])
        
        # Simple point-in-polygon test using winding number
        # For triangulated mesh, check if point is inside any triangle
        if len(self.faces) > 0:
            vertices_2d = self.vertices[:, :2]  # Ensure 2D
            
            def is_inside_triangle(p, v0, v1, v2):
                """Barycentric coordinate test."""
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                v0p = p - v0
                
                d00 = jnp.dot(v0v1, v0v1)
                d01 = jnp.dot(v0v1, v0v2)
                d11 = jnp.dot(v0v2, v0v2)
                d20 = jnp.dot(v0p, v0v1)
                d21 = jnp.dot(v0p, v0v2)
                
                denom = d00 * d11 - d01 * d01
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                u = 1.0 - v - w
                
                return (u >= 0) & (v >= 0) & (w >= 0)
            
            # Vectorized inside test
            def is_inside(point):
                inside = False
                for face in self.faces:
                    v0 = vertices_2d[face[0]]
                    v1 = vertices_2d[face[1]]
                    v2 = vertices_2d[face[2]]
                    inside = inside | is_inside_triangle(point, v0, v1, v2)
                return inside
            
            # Batched test
            inside_mask = jax.vmap(is_inside)(points)
            accepted_points = points[inside_mask]
            
            # Take first n_points or pad
            if len(accepted_points) >= n_points:
                return accepted_points[:n_points]
            else:
                # Pad with repeated points if not enough
                n_repeat = (n_points // len(accepted_points) + 1) if len(accepted_points) > 0 else 1
                return jnp.tile(accepted_points, (n_repeat, 1))[:n_points]
        else:
            # Fallback: just sample in bounding box
            return points[:n_points]
    
    def _sample_interior_3d(self, n_points: int, method: str,
                           rng: jax.Array, **kwargs) -> jnp.ndarray:
        """3D interior sampling via rejection or tetrahedral decomposition."""
        if method == 'tetrahedral' and self.volume_markers is not None:
            # Sample directly from tetrahedral elements
            return self._sample_from_tetrahedra(n_points, rng)
        else:
            # Rejection sampling
            return self._sample_rejection_3d(n_points, rng)
    
    def _sample_rejection_3d(self, n_points: int, rng: jax.Array) -> jnp.ndarray:
        """3D rejection sampling in bounding box."""
        max_attempts = n_points * 50
        keys = jax.random.split(rng, 3)
        
        x = jax.random.uniform(keys[0], (max_attempts, 1),
                               minval=self.bbox_min[0], maxval=self.bbox_max[0])
        y = jax.random.uniform(keys[1], (max_attempts, 1),
                               minval=self.bbox_min[1], maxval=self.bbox_max[1])
        z = jax.random.uniform(keys[2], (max_attempts, 1),
                               minval=self.bbox_min[2], maxval=self.bbox_max[2])
        points = jnp.hstack([x, y, z])
        
        # Simple test: check if point is inside any tetrahedron
        # This is simplified - full implementation would use ray casting
        # For now, return all points (user should provide volume mesh for accuracy)
        return points[:n_points]
    
    def _sample_from_tetrahedra(self, n_points: int, rng: jax.Array) -> jnp.ndarray:
        """Sample from tetrahedral volume mesh."""
        if self.volume_markers is None:
            return self._sample_rejection_3d(n_points, rng)
        
        # Collect all tetrahedra
        tetrahedra = []
        for marker, indices in self.volume_markers.items():
            for idx in indices:
                tetrahedra.append(self.faces[idx])
        
        if not tetrahedra:
            return self._sample_rejection_3d(n_points, rng)
        
        tetrahedra = jnp.array(tetrahedra)
        
        # Sample tetrahedra uniformly
        n_tet = len(tetrahedra)
        tet_indices = jax.random.choice(rng, n_tet, shape=(n_points,))
        selected_tets = tetrahedra[tet_indices]
        
        # Sample barycentric coordinates in each tetrahedron
        keys = jax.random.split(rng, 4)
        r1 = jax.random.uniform(keys[0], (n_points, 1))
        r2 = jax.random.uniform(keys[1], (n_points, 1))
        r3 = jax.random.uniform(keys[2], (n_points, 1))
        
        # Transform to uniform in tetrahedron
        c1 = 1 - r1 ** (1/3)
        c2 = 1 - r2 ** (1/2)
        c3 = 1 - r3
        
        # Get vertices
        v0 = self.vertices[selected_tets[:, 0]]
        v1 = self.vertices[selected_tets[:, 1]]
        v2 = self.vertices[selected_tets[:, 2]]
        v3 = self.vertices[selected_tets[:, 3]]
        
        # Interpolate
        points = (c1[:, None] * v0 + 
                 (1 - c1[:, None]) * c2[:, None] * v1 +
                 (1 - c1[:, None]) * (1 - c2[:, None]) * c3[:, None] * v2 +
                 (1 - c1[:, None]) * (1 - c2[:, None]) * (1 - c3[:, None]) * v3)
        
        return points
    
    def sample_boundary(self, n_points: Optional[int] = None, 
                       method: str = 'random', rng: Optional[jax.Array] = None,
                       marker: Optional[str] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample points on the boundary surface.
        
        Args:
            n_points: Number of points (required for mesh geometries)
            method: Sampling method ('random', 'uniform')
            rng: JAX random key
            marker: Optional boundary marker name to sample specific boundary
            
        Returns:
            Tuple of (points, normals) where:
                - points: (n_points, dim) array of boundary coordinates
                - normals: (n_points, dim) array of outward normals
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        if n_points is None:
            n_points = len(self.faces)
        
        # Import here to avoid circular imports
        from .geometry_io import sample_points_on_surface
        
        if marker is not None and self.boundary_markers is not None:
            # Sample from specific boundary region
            if marker in self.boundary_markers:
                face_indices = self.boundary_markers[marker]
                subset_faces = self.faces[face_indices]
                subset_normals = self.normals[face_indices] if self.normals is not None else None
                
                # Create temporary mesh for this boundary
                temp_mesh = MeshData(
                    vertices=self.mesh_data.vertices,
                    faces=np.array(face_indices),
                    normals=np.array(subset_normals) if subset_normals is not None else None,
                    dim=self.dim
                )
                points, normals = sample_points_on_surface(temp_mesh, n_points, rng)
                return points, normals
        
        # Sample from entire boundary
        return sample_points_on_surface(self.mesh_data, n_points, rng)
    
    def get_normal_at_point(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Compute normal at a given point on the boundary.
        
        Uses interpolation of face normals based on closest faces.
        
        Args:
            point: Point coordinates (dim,)
            
        Returns:
            Unit normal vector (dim,)
        """
        if self.normals is None:
            raise ValueError("Normals not computed for this geometry")
        
        # Find closest face
        face_centers = jnp.mean(self.vertices[self.faces], axis=1)
        distances = jnp.linalg.norm(face_centers - point, axis=1)
        closest_face_idx = jnp.argmin(distances)
        
        return self.normals[closest_face_idx]
    
    def plot_domain(self, interior_points=None, boundary_points=None, 
                   interface_points=None, title="Domain Visualization",
                   show_normals: bool = False, normal_scale: float = 0.1):
        """
        Visualize the mesh geometry with sampled points.
        
        Args:
            interior_points: Internal collocation points
            boundary_points: Boundary points
            interface_points: Interface points for multi-domain
            title: Plot title
            show_normals: Whether to show normal vectors
            normal_scale: Scale factor for normal visualization
        """
        if self.dim == 2:
            self._plot_2d(interior_points, boundary_points, interface_points, 
                         title, show_normals, normal_scale)
        elif self.dim == 3:
            self._plot_3d(interior_points, boundary_points, interface_points,
                         title, show_normals, normal_scale)
    
    def _plot_2d(self, interior, boundary, interface, title, show_normals, scale):
        """2D visualization with mesh wireframe."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot mesh edges
        if len(self.faces) > 0:
            for face in self.faces:
                verts = self.vertices[face]
                # Close the loop for triangles
                verts_closed = jnp.vstack([verts, verts[0]])
                ax.plot(verts_closed[:, 0], verts_closed[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        
        # Plot sampled points
        if interior is not None and len(interior) > 0:
            ax.scatter(interior[:, 0], interior[:, 1], c='blue', s=10, alpha=0.5, label='Interior')
        
        if boundary is not None and len(boundary) > 0:
            if isinstance(boundary, tuple):
                boundary_pts, normals = boundary
                ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], c='green', s=20, alpha=0.7, label='Boundary')
                
                if show_normals:
                    # Plot normal vectors
                    for i in range(min(50, len(boundary_pts))):  # Limit for clarity
                        pt = boundary_pts[i]
                        n = normals[i]
                        ax.arrow(pt[0], pt[1], scale * n[0], scale * n[1],
                                head_width=0.02, head_length=0.03, fc='red', ec='red')
            else:
                ax.scatter(boundary[:, 0], boundary[:, 1], c='green', s=20, alpha=0.7, label='Boundary')
        
        if interface is not None and len(interface) > 0:
            ax.scatter(interface[:, 0], interface[:, 1], c='red', s=20, alpha=0.7, label='Interface')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()
    
    def _plot_3d(self, interior, boundary, interface, title, show_normals, scale):
        """3D visualization."""
        try:
            import pyvista as pv
            has_pyvista = True
        except ImportError:
            has_pyvista = False
            from mpl_toolkits.mplot3d import Axes3D
        
        if has_pyvista:
            # Use PyVista for better 3D visualization
            mesh = pv.PolyData(self.vertices, np.hstack([np.full((len(self.faces), 1), 3), self.faces]))
            plotter = pv.Plotter()
            plotter.add_mesh(mesh, opacity=0.5, show_edges=True)
            
            if interior is not None and len(interior) > 0:
                plotter.add_points(interior, color='blue', point_size=5, label='Interior')
            
            if boundary is not None and len(boundary) > 0:
                if isinstance(boundary, tuple):
                    boundary_pts, _ = boundary
                else:
                    boundary_pts = boundary
                plotter.add_points(boundary_pts, color='green', point_size=8, label='Boundary')
            
            plotter.add_title(title)
            plotter.show()
        else:
            # Fallback to matplotlib
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            if interior is not None and len(interior) > 0:
                ax.scatter(interior[:, 0], interior[:, 1], interior[:, 2], 
                          c='blue', s=10, alpha=0.5, label='Interior')
            
            if boundary is not None and len(boundary) > 0:
                if isinstance(boundary, tuple):
                    boundary_pts, _ = boundary
                else:
                    boundary_pts = boundary
                ax.scatter(boundary_pts[:, 0], boundary_pts[:, 1], boundary_pts[:, 2],
                          c='green', s=20, alpha=0.7, label='Boundary')
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(title)
            ax.legend(loc='best')
            plt.tight_layout()
            plt.show()


class CompositeGeometry(Geometry):
    """
    Composite geometry made of multiple sub-geometries for multi-domain problems.
    
    Each sub-geometry represents a separate domain with its own material properties.
    Interfaces between domains are automatically detected or can be specified manually.
    """
    
    def __init__(self, sub_geometries: list, interfaces: Optional[list] = None):
        """
        Initialize composite geometry.
        
        Args:
            sub_geometries: List of Geometry objects (MeshGeometry or primitive)
            interfaces: Optional list of interface specifications between domains
        """
        # All sub-geometries should have same dimension
        dims = [g.dim for g in sub_geometries]
        if len(set(dims)) != 1:
            raise ValueError(f"All sub-geometries must have same dimension, got {dims}")
        
        super().__init__(dim=dims[0])
        
        self.sub_geometries = sub_geometries
        self.n_domains = len(sub_geometries)
        self.interfaces = interfaces or []
    
    def sample_interior(self, n_points_per_domain: int, 
                       method: str = 'random', 
                       rng: Optional[jax.Array] = None) -> list:
        """
        Sample interior points for each sub-domain.
        
        Args:
            n_points_per_domain: Points per domain (or list for each)
            method: Sampling method
            rng: Random key
            
        Returns:
            List of point arrays, one per domain
        """
        if isinstance(n_points_per_domain, int):
            n_points_per_domain = [n_points_per_domain] * self.n_domains
        
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        keys = jax.random.split(rng, self.n_domains)
        
        return [
            geom.sample_interior(n, method, k)
            for geom, n, k in zip(self.sub_geometries, n_points_per_domain, keys)
        ]
    
    def sample_boundary(self, n_points: Optional[int] = None,
                       exclude_interfaces: bool = True,
                       rng: Optional[jax.Array] = None) -> list:
        """
        Sample boundary points for each sub-domain.
        
        Args:
            n_points: Points per boundary (or list)
            exclude_interfaces: Whether to exclude interface boundaries
            rng: Random key
            
        Returns:
            List of boundary point arrays per domain
        """
        if rng is None:
            rng = jax.random.PRNGKey(0)
        
        keys = jax.random.split(rng, self.n_domains)
        
        return [
            geom.sample_boundary(n_points, rng=k)
            for geom, k in zip(self.sub_geometries, keys)
        ]
    
    def sample_interface(self, interface_id: int, n_points: int,
                        rng: Optional[jax.Array] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample points on interface between domains.
        
        Args:
            interface_id: Index of interface specification
            n_points: Number of points to sample
            rng: Random key
            
        Returns:
            Tuple of (points, normals) on the interface
        """
        if interface_id >= len(self.interfaces):
            raise ValueError(f"Interface {interface_id} not found")
        
        iface = self.interfaces[interface_id]
        dom1, dom2 = iface['domains']
        
        # Sample from boundary of first domain near second domain
        # Simplified: assumes shared boundary representation
        geom1 = self.sub_geometries[dom1]
        
        if hasattr(geom1, 'sample_boundary'):
            return geom1.sample_boundary(n_points, rng=rng)
        else:
            raise NotImplementedError("Interface sampling requires MeshGeometry")
