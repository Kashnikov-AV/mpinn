"""
Geometry I/O module for loading complex geometries from files.

Supports formats:
- STL (stereolithography) - surface mesh
- OBJ (Wavefront) - surface mesh with normals
- Gmsh (.msh) - volume mesh with boundary markers
- JSON custom format - simple geometries
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import struct


@dataclass
class MeshData:
    """Container for mesh data loaded from file."""
    vertices: np.ndarray  # (N, dim) array of vertex coordinates
    faces: np.ndarray     # (M, 3) or (M, 4) array of face indices
    normals: Optional[np.ndarray] = None  # (M, 3) face normals
    boundary_markers: Optional[Dict[str, np.ndarray]] = None  # boundary face groups
    volume_markers: Optional[Dict[str, np.ndarray]] = None  # volume element groups
    dim: int = 3  # dimension of the geometry


def load_stl(filepath: str, compute_normals: bool = True) -> MeshData:
    """
    Load geometry from STL file (binary or ASCII).
    
    Args:
        filepath: Path to .stl file
        compute_normals: Whether to compute face normals automatically
        
    Returns:
        MeshData with vertices and faces
    """
    with open(filepath, 'rb') as f:
        header = f.read(80)
        n_triangles = struct.unpack('<I', f.read(4))[0]
        
        # Check if binary STL
        expected_size = 84 + n_triangles * 50
        f.seek(0, 2)  # go to end
        actual_size = f.tell()
        f.seek(84)
        
        vertices = []
        faces = []
        normals = []
        
        if actual_size == expected_size:
            # Binary STL
            for i in range(n_triangles):
                normal = struct.unpack('<3f', f.read(12))
                v1 = struct.unpack('<3f', f.read(12))
                v2 = struct.unpack('<3f', f.read(12))
                v3 = struct.unpack('<3f', f.read(12))
                attr = f.read(2)  # attribute byte count
                
                normals.append(normal)
                idx_base = len(vertices)
                vertices.extend([v1, v2, v3])
                faces.append([idx_base, idx_base + 1, idx_base + 2])
        else:
            # ASCII STL
            f.seek(0)
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].decode('ascii').strip().lower()
                if line.startswith('facet normal'):
                    parts = line.split()
                    normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                    normals.append(normal)
                    
                    i += 1  # skip 'outer loop'
                    triangle_verts = []
                    for j in range(3):
                        i += 1
                        vert_line = lines[i].decode('ascii').strip()
                        if vert_line.startswith('vertex'):
                            parts = vert_line.split()
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            triangle_verts.append(vertex)
                    
                    idx_base = len(vertices)
                    vertices.extend(triangle_verts)
                    faces.append([idx_base, idx_base + 1, idx_base + 2])
                    i += 2  # skip 'endloop' and 'endfacet'
                else:
                    i += 1
        
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)
        normals = np.array(normals, dtype=np.float64) if compute_normals else None
        
        return MeshData(
            vertices=vertices,
            faces=faces,
            normals=normals,
            dim=3
        )


def load_obj(filepath: str, compute_normals: bool = True) -> MeshData:
    """
    Load geometry from Wavefront OBJ file.
    
    Args:
        filepath: Path to .obj file
        compute_normals: Whether to compute face normals if not present
        
    Returns:
        MeshData with vertices, faces, and optionally normals
    """
    vertices = []
    faces = []
    vertex_normals = []
    face_normal_indices = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':  # vertex
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif parts[0] == 'vn':  # vertex normal
                normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertex_normals.append(normal)
            elif parts[0] == 'f':  # face
                face_verts = []
                face_norms = []
                for p in parts[1:]:
                    indices = p.split('/')
                    vert_idx = int(indices[0]) - 1  # OBJ uses 1-based indexing
                    face_verts.append(vert_idx)
                    if len(indices) >= 3 and indices[2]:
                        norm_idx = int(indices[2]) - 1
                        face_norms.append(norm_idx)
                
                # Triangulate if quad or higher
                if len(face_verts) > 3:
                    for i in range(1, len(face_verts) - 1):
                        faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                        if face_norms:
                            face_normal_indices.append([face_norms[0], face_norms[i], face_norms[i + 1]])
                else:
                    faces.append(face_verts[:3])
                    if face_norms:
                        face_normal_indices.append(face_norms[:3])
    
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)
    
    # Compute face normals if needed
    normals = None
    if compute_normals:
        if vertex_normals and face_normal_indices:
            # Use pre-computed vertex normals
            normals = np.zeros((len(faces), 3), dtype=np.float64)
            for i, fn_indices in enumerate(face_normal_indices):
                face_normal = np.mean([vertex_normals[idx] for idx in fn_indices], axis=0)
                normals[i] = face_normal / (np.linalg.norm(face_normal) + 1e-10)
        else:
            # Compute face normals from geometry
            normals = np.zeros((len(faces), 3), dtype=np.float64)
            for i, face in enumerate(faces):
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(normal)
                if norm_len > 1e-10:
                    normals[i] = normal / norm_len
    
    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        dim=3
    )


def load_gmsh(filepath: str) -> MeshData:
    """
    Load geometry from Gmsh .msh file (version 2.x or 4.x).
    
    Args:
        filepath: Path to .msh file
        
    Returns:
        MeshData with vertices, elements, and boundary markers
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    vertices = []
    elements = {}  # elem_type -> list of (node_ids, marker)
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == '$Nodes':
            i += 1
            n_nodes = int(lines[i].strip())
            i += 1
            for _ in range(n_nodes):
                parts = lines[i].strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
                i += 1
        
        elif line == '$Elements' or line == '$Elements2':
            i += 1
            n_elements = int(lines[i].strip())
            i += 1
            
            # Check Gmsh version by format
            first_elem_line = lines[i].strip().split()
            
            if len(first_elem_line) >= 4:
                # Version 2.x format: tag, node1, node2, node3...
                for _ in range(n_elements):
                    parts = lines[i].strip().split()
                    elem_tag = int(parts[0])
                    elem_type = int(parts[1])
                    n_tags = int(parts[2])
                    
                    # Physical marker is the first tag
                    physical_marker = int(parts[3]) if n_tags >= 1 else 0
                    
                    node_ids = [int(x) - 1 for x in parts[3 + n_tags:]]  # 1-based to 0-based
                    
                    if elem_type not in elements:
                        elements[elem_type] = []
                    elements[elem_type].append((node_ids, physical_marker))
                    i += 1
            else:
                # Version 4.x format (more complex, simplified parsing)
                for _ in range(n_elements):
                    parts = lines[i].strip().split()
                    elem_type = int(parts[0])
                    num_tags = int(parts[1])
                    physical_marker = int(parts[2]) if num_tags >= 1 else 0
                    
                    node_ids = [int(x) - 1 for x in parts[2 + num_tags:]]
                    
                    if elem_type not in elements:
                        elements[elem_type] = []
                    elements[elem_type].append((node_ids, physical_marker))
                    i += 1
        else:
            i += 1
    
    vertices = np.array(vertices, dtype=np.float64)
    
    # Separate different element types
    # Type 1: line (2 nodes), Type 2: triangle (3 nodes), Type 4: tetrahedron (4 nodes)
    faces_2d = []  # triangles for surface
    volumes_3d = []  # tetrahedra for volume
    boundary_markers = {}
    volume_markers = {}
    
    for elem_type, elem_list in elements.items():
        if elem_type == 2:  # triangles
            for nodes, marker in elem_list:
                if len(nodes) == 3:
                    faces_2d.append(nodes)
                    if marker not in boundary_markers:
                        boundary_markers[marker] = []
                    boundary_markers[marker].append(len(faces_2d) - 1)
        elif elem_type == 4:  # tetrahedra
            for nodes, marker in elem_list:
                if len(nodes) == 4:
                    volumes_3d.append(nodes)
                    if marker not in volume_markers:
                        volume_markers[marker] = []
                    volume_markers[marker].append(len(volumes_3d) - 1)
    
    faces = np.array(faces_2d, dtype=np.int32) if faces_2d else np.empty((0, 3), dtype=np.int32)
    
    # Determine dimension
    dim = 2 if len(volumes_3d) == 0 else 3
    
    return MeshData(
        vertices=vertices,
        faces=faces,
        dim=dim,
        boundary_markers=boundary_markers if boundary_markers else None,
        volume_markers=volume_markers if volume_markers else None
    )


def load_geometry(filepath: str, file_format: Optional[str] = None, **kwargs) -> MeshData:
    """
    Auto-detect file format and load geometry.
    
    Args:
        filepath: Path to geometry file
        file_format: Optional format override ('stl', 'obj', 'gmsh')
        **kwargs: Additional arguments passed to specific loaders
        
    Returns:
        MeshData object
    """
    if file_format is None:
        ext = filepath.lower().split('.')[-1]
        file_format = ext
    
    if file_format == 'stl':
        return load_stl(filepath, **kwargs)
    elif file_format == 'obj':
        return load_obj(filepath, **kwargs)
    elif file_format == 'msh':
        return load_gmsh(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported: stl, obj, msh")


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute face normals for a mesh.
    
    Args:
        vertices: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices
        
    Returns:
        (M, 3) array of unit face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    normals = np.cross(edge1, edge2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    
    return normals / norms


def sample_points_on_surface(
    mesh: MeshData, 
    n_points: int, 
    rng: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample points uniformly on mesh surface with corresponding normals.
    
    Args:
        mesh: MeshData object
        n_points: Number of points to sample
        rng: JAX random key
        
    Returns:
        Tuple of (points, normals) where:
            - points: (n_points, 3) array of sampled coordinates
            - normals: (n_points, 3) array of interpolated normals
    """
    if rng is None:
        rng = jnp.array([0, 0])
    
    vertices = jnp.array(mesh.vertices, dtype=jnp.float64)
    faces = jnp.array(mesh.faces, dtype=jnp.int32)
    normals = jnp.array(mesh.normals, dtype=jnp.float64) if mesh.normals is not None else None
    
    # Compute face areas for weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    edge1 = v1 - v0
    edge2 = v2 - v0
    cross_prod = jnp.cross(edge1, edge2)
    face_areas = 0.5 * jnp.linalg.norm(cross_prod, axis=1)
    
    # Sample faces proportionally to area
    probs = face_areas / jnp.sum(face_areas)
    face_indices = jax.random.choice(rng, faces.shape[0], shape=(n_points,), p=probs)
    
    # Generate barycentric coordinates
    rng1, rng2 = jax.random.split(rng, 2)
    u1 = jax.random.uniform(rng1, (n_points,))
    u2 = jax.random.uniform(rng2, (n_points,))
    
    # Ensure points are within triangle
    mask = u1 + u2 > 1.0
    u1 = jnp.where(mask, 1.0 - u1, u1)
    u2 = jnp.where(mask, 1.0 - u2, u2)
    
    # Compute point positions
    selected_faces = faces[face_indices]
    v0_sel = vertices[selected_faces[:, 0]]
    v1_sel = vertices[selected_faces[:, 1]]
    v2_sel = vertices[selected_faces[:, 2]]
    
    points = v0_sel + u1[:, None] * (v1_sel - v0_sel) + u2[:, None] * (v2_sel - v0_sel)
    
    # Interpolate normals
    if normals is not None:
        n0_sel = normals[selected_faces[:, 0]]
        n1_sel = normals[selected_faces[:, 1]]
        n2_sel = normals[selected_faces[:, 2]]
        sampled_normals = n0_sel + u1[:, None] * (n1_sel - n0_sel) + u2[:, None] * (n2_sel - n0_sel)
        sampled_normals = sampled_normals / (jnp.linalg.norm(sampled_normals, axis=1, keepdims=True) + 1e-10)
    else:
        # Use face normals
        sampled_normals = normals[face_indices]
    
    return points, sampled_normals


# Import jax here to avoid circular imports
import jax
import jax.numpy as jnp
