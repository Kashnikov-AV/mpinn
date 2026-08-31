from . import (
    analytic,
    bc,
    bc_2d_3d,
    config,
    geom,
    geometry_io,
    multidomain,
    pde,
    pde_2d_3d,
    pinn_core,
    plotting,
    runner,
    sources,
    weight_strategies,
)

# Экспорт новых классов геометрий
from .geom_extended import (
    Annulus2D,
    GeometryBase,
    HollowCylinder3D,
    HollowSphere3D,
    MeshGeometry,
)

__all__ = [
    "Annulus2D",
    # Новые классы
    "GeometryBase",
    "HollowCylinder3D",
    "HollowSphere3D",
    "MeshGeometry",
    "analytic",
    "bc",
    "bc_2d_3d",
    "config",
    "geom",
    "geom_extended",
    "geometry_io",
    "multidomain",
    "pde",
    "pde_2d_3d",
    "pinn_core",
    "plotting",
    "runner",
    "sources",
    "weight_strategies",
]
