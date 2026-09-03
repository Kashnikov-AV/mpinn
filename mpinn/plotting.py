"""
Модуль визуализации для PINN.
Поддерживает 1D, 2D и 3D графики, а также историю обучения.
Работает с JAX и NumPy массивами напрямую.
"""

import os
from typing import Optional, List, Dict, Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .geom import GeometryBase

# ===================== 1D графики (обратно совместимые) =====================
def show_plot(
    x_test: Union[np.ndarray, "jnp.ndarray"],
    T_pred: Union[np.ndarray, "jnp.ndarray"],
    T_exact: Union[np.ndarray, "jnp.ndarray"],
    title: str = "Сравнение PINN и точного решения"
) -> None:
    """Отображает 1D график сравнения."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, T_exact, "b-", label="Аналитическое решение", linewidth=2)
    plt.plot(x_test, T_pred, "r:", label="ФИНС", linewidth=6)
    plt.title(title)
    plt.xlabel("x, м")
    plt.ylabel("T, К")
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_test.min(), x_test.max())
    plt.tight_layout()
    plt.show()


def save_plot(
    x_test: Union[np.ndarray, "jnp.ndarray"],
    T_pred: Union[np.ndarray, "jnp.ndarray"],
    T_exact: Union[np.ndarray, "jnp.ndarray"],
    save_path: str = "",
    title: str = "Сравнение PINN и точного решения"
) -> None:
    """Сохраняет 1D график сравнения."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, T_exact, "b-", label="Аналитическое решение", linewidth=2)
    plt.plot(x_test, T_pred, "r:", label="ФИНС", linewidth=6)
    plt.title(title)
    plt.xlabel("x, м")
    plt.ylabel("T, К")
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_test.min(), x_test.max())
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=72, bbox_inches="tight")
    plt.close()


def show_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """Отображает историю обучения."""
    steps = history.get("steps", [])
    if not steps:
        print("Нет данных для отображения истории.")
        return

    plt.figure(figsize=(12, 7))

    if "pde" in history:
        plt.semilogy(steps, history["pde"], "b", label="PDE", linewidth=2)

    if "total_loss" in history:
        plt.semilogy(
            steps, history["total_loss"], "g--", label="Общая", linewidth=2, alpha=0.7
        )

    bc_keys = [k for k in history.keys() if k.startswith("bc_")]
    colors = ["r", "c", "m", "y", "k", "orange", "purple"]
    for i, key in enumerate(bc_keys):
        color = colors[i % len(colors)]
        label = key.replace("bc_", "ГУ ")
        plt.semilogy(
            steps,
            history[key],
            f"{color}--",
            label=label,
            linewidth=2,
            alpha=0.7,
        )

    plt.title("История обучения")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
        print(f"График сохранён: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


# ===================== 2D/3D графики =====================
def plot_2d_contour(
    geom: "GeometryBase",
    predict_fn: Callable[[np.ndarray], np.ndarray],
    resolution: int = 50,
    title: str = "Температурное поле",
    xlabel: str = "x, м",
    ylabel: str = "y, м",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Строит контурный график температуры на 2D области."""
    if geom.dim != 2:
        raise ValueError("plot_2d_contour поддерживает только 2D геометрию.")

    if hasattr(geom, "x_min") and hasattr(geom, "x_max"):
        x_min, x_max = geom.x_min, geom.x_max
        y_min, y_max = geom.y_min, geom.y_max
    else:
        try:
            min_corner, max_corner = geom.bounds
            x_min, y_min = min_corner[0], min_corner[1]
            x_max, y_max = max_corner[0], max_corner[1]
        except AttributeError:
            raise ValueError("Не удалось определить границы геометрии.")

    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel()])

    T = predict_fn(points).reshape(resolution, resolution)

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, T, levels=50, cmap="jet")
    plt.colorbar(contour, label="T, К")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_slices(
    geom: "GeometryBase",
    predict_fn: Callable[[np.ndarray], np.ndarray],
    slice_planes: Optional[List[Dict[str, float]]] = None,
    resolution: int = 30,
    title: str = "Срезы температуры",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Строит срезы температуры в 3D по плоскостям."""
    if geom.dim != 3:
        raise ValueError("plot_3d_slices поддерживает только 3D геометрию.")

    if hasattr(geom, "x_min") and hasattr(geom, "x_max"):
        x_min, x_max = geom.x_min, geom.x_max
        y_min, y_max = geom.y_min, geom.y_max
        z_min, z_max = geom.z_min, geom.z_max
    else:
        try:
            min_corner, max_corner = geom.bounds
            x_min, y_min, z_min = min_corner[0], min_corner[1], min_corner[2]
            x_max, y_max, z_max = max_corner[0], max_corner[1], max_corner[2]
        except AttributeError:
            raise ValueError("Не удалось определить границы геометрии.")

    if slice_planes is None:
        slice_planes = [
            {"axis": "x", "value": (x_min + x_max) / 2},
            {"axis": "y", "value": (y_min + y_max) / 2},
            {"axis": "z", "value": (z_min + z_max) / 2},
        ]

    n_planes = len(slice_planes)
    fig, axes = plt.subplots(1, n_planes, figsize=(5 * n_planes, 4))
    if n_planes == 1:
        axes = [axes]

    for ax, plane in zip(axes, slice_planes):
        axis = plane["axis"]
        value = plane["value"]

        if axis == "x":
            xs = np.full(resolution * resolution, value)
            y = np.linspace(y_min, y_max, resolution)
            z = np.linspace(z_min, z_max, resolution)
            Y, Z = np.meshgrid(y, z, indexing="ij")
            points = np.column_stack([xs.ravel(), Y.ravel(), Z.ravel()])
            T_slice = predict_fn(points).reshape(resolution, resolution)
            contour = ax.contourf(Y, Z, T_slice, levels=50, cmap="jet")
            ax.set_xlabel("y, м")
            ax.set_ylabel("z, м")
            ax.set_title(f"x = {value:.2f}")
        elif axis == "y":
            ys = np.full(resolution * resolution, value)
            x = np.linspace(x_min, x_max, resolution)
            z = np.linspace(z_min, z_max, resolution)
            X, Z = np.meshgrid(x, z, indexing="ij")
            points = np.column_stack([X.ravel(), ys.ravel(), Z.ravel()])
            T_slice = predict_fn(points).reshape(resolution, resolution)
            contour = ax.contourf(X, Z, T_slice, levels=50, cmap="jet")
            ax.set_xlabel("x, м")
            ax.set_ylabel("z, м")
            ax.set_title(f"y = {value:.2f}")
        elif axis == "z":
            zs = np.full(resolution * resolution, value)
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x, y, indexing="ij")
            points = np.column_stack([X.ravel(), Y.ravel(), zs.ravel()])
            T_slice = predict_fn(points).reshape(resolution, resolution)
            contour = ax.contourf(X, Y, T_slice, levels=50, cmap="jet")
            ax.set_xlabel("x, м")
            ax.set_ylabel("y, м")
            ax.set_title(f"z = {value:.2f}")
        else:
            raise ValueError(f"Неизвестная ось: {axis}")

        plt.colorbar(contour, ax=ax, label="T, К")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=72, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_3d_surface_plotly(
    X: Union[np.ndarray, "jnp.ndarray"],
    Y: Union[np.ndarray, "jnp.ndarray"],
    Z: Union[np.ndarray, "jnp.ndarray"],
    title: str = "3D Температурное поле",
    labels: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Создает интерактивный 3D график поверхности с помощью Plotly.
    """
    if not HAS_PLOTLY:
        raise ImportError(
            "Plotly не установлен. Установите его: pip install plotly"
        )

    # Приводим к NumPy для надёжности (plotly лучше работает с numpy)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    if labels is None:
        labels = {'x': 'X', 'y': 'Y', 'z': 'Температура'}

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='jet')])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=labels.get('x', 'X'),
            yaxis_title=labels.get('y', 'Y'),
            zaxis_title=labels.get('z', 'Z'),
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path)
        print(f"График сохранён: {save_path}")

    if show:
        fig.show()