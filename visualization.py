"""
Visualization module for optimization functions.

This module contains functions for plotting and visualizing optimization
benchmark functions in both 3D and contour formats using either matplotlib or plotly.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def plot_function_3d_matplotlib(
    func, x_range, y_range, title, optimal_point, levels=None, save_figs=False
):
    """
    Plot 3D visualization with 4 different views using matplotlib

    Args:
        func: function to plot
        x_range: tuple of (min, max) for x axis
        y_range: tuple of (min, max) for y axis
        title: title for the plots
        optimal_point: tuple of (x, y) optimal coordinates
        levels: contour levels for contour plot
        save_figs: whether to save figures to figs folder
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(12, 6))
    opt_x, opt_y = optimal_point
    opt_z = func(opt_x, opt_y)

    # Top view (looking down from above, elev=90, azim=0) - Orthographic
    ax1 = fig.add_subplot(221, projection="3d")
    surf1 = ax1.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax1.view_init(elev=90, azim=0)
    ax1.set_proj_type("ortho")  # Orthographic projection
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("f(x, y)")
    ax1.set_title(f"{title} - Top View ")
    ax1.scatter(
        [opt_x],
        [opt_y],
        [opt_z],
        color="red",
        s=100,
        label=f"Optimal ({opt_x},{opt_y})",
    )

    # Front view (elev=0, azim=0) - Orthographic
    ax2 = fig.add_subplot(222, projection="3d")
    surf2 = ax2.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax2.view_init(elev=0, azim=0)
    ax2.set_proj_type("ortho")  # Orthographic projection
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("f(x, y)")
    ax2.set_title(f"{title} - Front View")
    ax2.scatter(
        [opt_x],
        [opt_y],
        [opt_z],
        color="red",
        s=100,
        label=f"Optimal ({opt_x},{opt_y})",
    )

    # Side view (elev=0, azim=90) - Orthographic
    ax3 = fig.add_subplot(223, projection="3d")
    surf3 = ax3.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax3.view_init(elev=0, azim=90)
    ax3.set_proj_type("ortho")  # Orthographic projection
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("f(x, y)")
    ax3.set_title(f"{title} - Side View")
    ax3.scatter(
        [opt_x],
        [opt_y],
        [opt_z],
        color="red",
        s=100,
        label=f"Optimal ({opt_x},{opt_y})",
    )

    # Perspective view (elev=30, azim=45) - Perspective projection (default)
    ax4 = fig.add_subplot(224, projection="3d")
    surf4 = ax4.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8)
    ax4.view_init(elev=30, azim=45)
    # ax4.set_proj_type('persp')  # Perspective projection (default, no need to set)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("f(x, y)")
    ax4.set_title(f"{title} - Perspective View")
    ax4.scatter(
        [opt_x],
        [opt_y],
        [opt_z],
        color="red",
        s=100,
        label=f"Optimal ({opt_x},{opt_y})",
    )

    # Adjust subplot layout first to make room for colorbar
    plt.subplots_adjust(
        left=0.05, right=0.75, top=0.95, bottom=0.05, wspace=0.25, hspace=0.3
    )

    # Add colorbar to the figure with proper positioning
    cbar = fig.colorbar(
        surf4, ax=[ax1, ax2, ax3, ax4], shrink=0.7, pad=0.15, fraction=0.04, aspect=25
    )
    
    # Save figure if requested
    if save_figs:
        # Create figs directory if it doesn't exist
        os.makedirs('figs', exist_ok=True)
        # Clean filename from title
        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        filename = f'figs/{clean_title}_3D.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved 3D plot: {filename}")
    
    plt.show()

    # Show separate contour plot
    plot_contour_matplotlib(func, x_range, y_range, title, optimal_point, levels, save_figs)


def plot_contour_matplotlib(func, x_range, y_range, title, optimal_point, levels=None, save_figs=False):
    """
    Plot contour visualization of a function using matplotlib

    Args:
        func: function to plot
        x_range: tuple of (min, max) for x axis
        y_range: tuple of (min, max) for y axis
        title: title for the plot
        optimal_point: tuple of (x, y) optimal coordinates
        levels: contour levels, if None uses default 20 levels
        save_figs: whether to save figures to figs folder
    """
    x = np.linspace(x_range[0], x_range[1], 200)
    y = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    plt.figure(figsize=(8, 6))

    if levels is None:
        contour = plt.contour(X, Y, Z, levels=20, cmap="viridis")
    else:
        contour = plt.contour(X, Y, Z, levels=levels, cmap="viridis")

    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title(f"{title} - Contour Plot", fontsize=14)

    # Plot optimal solution point
    opt_x, opt_y = optimal_point
    plt.plot(opt_x, opt_y, "r*", markersize=20, label=f"Optimal ({opt_x},{opt_y})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.colorbar(contour, label="f(x, y)")

    plt.tight_layout()
    
    # Save figure if requested
    if save_figs:
        # Create figs directory if it doesn't exist
        os.makedirs('figs', exist_ok=True)
        # Clean filename from title
        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        filename = f'figs/{clean_title}_Contour.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved contour plot: {filename}")
    
    plt.show()


def plot_function_3d_plotly(func, x_range, y_range, title, optimal_point, levels=None, save_figs=False):
    """
    Plot 3D visualization with 4 different views using plotly

    Args:
        func: function to plot
        x_range: tuple of (min, max) for x axis
        y_range: tuple of (min, max) for y axis
        title: title for the plots
        optimal_point: tuple of (x, y) optimal coordinates
        levels: contour levels for contour plot
        save_figs: whether to save figures to figs folder
    """
    x = np.linspace(x_range[0], x_range[1], 50)  # Reduced resolution for plotly
    y = np.linspace(y_range[0], y_range[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    opt_x, opt_y = optimal_point
    opt_z = func(opt_x, opt_y)

    # Create subplots with 3D scenes
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=("Top View", "Front View", "Side View", "Perspective View"),
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    # Surface plot
    surface = go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", showscale=True)

    # Optimal point
    optimal_point_trace = go.Scatter3d(
        x=[opt_x],
        y=[opt_y],
        z=[opt_z],
        mode="markers",
        marker=dict(size=8, color="red"),
        name=f"Optimal ({opt_x},{opt_y})",
    )

    # Add surface and optimal point to each subplot
    for row in range(1, 3):
        for col in range(1, 3):
            fig.add_trace(surface, row=row, col=col)
            fig.add_trace(optimal_point_trace, row=row, col=col)

    # Set camera angles for different views
    camera_configs = [
        dict(eye=dict(x=0, y=0, z=2.5)),  # Top view
        dict(eye=dict(x=0, y=-2.5, z=0)),  # Front view
        dict(eye=dict(x=2.5, y=0, z=0)),  # Side view
        dict(eye=dict(x=1.25, y=1.25, z=1.25)),  # Perspective view
    ]

    scene_idx = 0
    for row in range(1, 3):
        for col in range(1, 3):
            scene_name = f"scene{scene_idx + 1}" if scene_idx > 0 else "scene"
            fig.update_layout(
                **{
                    scene_name: dict(
                        camera=camera_configs[scene_idx],
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="f(x, y)",
                    )
                }
            )
            scene_idx += 1

    fig.update_layout(
        title=f"{title} - 3D Views", height=500, width=700, showlegend=False
    )

    # Save figure if requested
    if save_figs:
        # Create figs directory if it doesn't exist
        os.makedirs('figs', exist_ok=True)
        # Clean filename from title
        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        filename = f'figs/{clean_title}_3D.html'
        fig.write_html(filename)
        print(f"Saved 3D plot: {filename}")

    fig.show()

    # Show separate contour plot
    plot_contour_plotly(func, x_range, y_range, title, optimal_point, levels, save_figs)


def plot_contour_plotly(func, x_range, y_range, title, optimal_point, levels=None, save_figs=False):
    """
    Plot contour visualization of a function using plotly

    Args:
        func: function to plot
        x_range: tuple of (min, max) for x axis
        y_range: tuple of (min, max) for y axis
        title: title for the plot
        optimal_point: tuple of (x, y) optimal coordinates
        levels: contour levels, if None uses default levels
        save_figs: whether to save figures to figs folder
    """
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    opt_x, opt_y = optimal_point

    fig = go.Figure()

    # Contour plot
    contour = go.Contour(
        x=x,
        y=y,
        z=Z,
        colorscale="Viridis",
        contours=dict(
            start=Z.min(),
            end=Z.max(),
            size=(Z.max() - Z.min()) / 20 if levels is None else None,
        )
        if levels is None
        else dict(
            start=min(levels) if hasattr(levels, "__iter__") else Z.min(),
            end=max(levels) if hasattr(levels, "__iter__") else Z.max(),
            size=(max(levels) - min(levels)) / len(levels)
            if hasattr(levels, "__iter__")
            else (Z.max() - Z.min()) / 20,
        ),
        line=dict(width=1),
        showscale=True,
    )

    fig.add_trace(contour)

    # Optimal point
    fig.add_trace(
        go.Scatter(
            x=[opt_x],
            y=[opt_y],
            mode="markers",
            marker=dict(size=15, color="red", symbol="star"),
            name=f"Optimal ({opt_x},{opt_y})",
        )
    )

    fig.update_layout(
        title=f"{title} - Contour Plot",
        xaxis_title="X",
        yaxis_title="Y",
        width=700,
        height=500,
        showlegend=True,
    )

    # Save figure if requested
    if save_figs:
        # Create figs directory if it doesn't exist
        os.makedirs('figs', exist_ok=True)
        # Clean filename from title
        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        filename = f'figs/{clean_title}_Contour.html'
        fig.write_html(filename)
        print(f"Saved contour plot: {filename}")

    fig.show()


def plot_function_3d(
    func, x_range, y_range, title, optimal_point, levels=None, backend="matplotlib", save_figs=False
):
    """
    Generic function to plot 3D visualization with selectable backend

    Args:
        func: function to plot
        x_range: tuple of (min, max) for x axis
        y_range: tuple of (min, max) for y axis
        title: title for the plots
        optimal_point: tuple of (x, y) optimal coordinates
        levels: contour levels for contour plot
        backend: 'matplotlib' or 'plotly'
        save_figs: whether to save figures to figs folder
    """
    if backend.lower() == "plotly":
        plot_function_3d_plotly(func, x_range, y_range, title, optimal_point, levels, save_figs)
    elif backend.lower() == "matplotlib":
        plot_function_3d_matplotlib(
            func, x_range, y_range, title, optimal_point, levels, save_figs
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'.")


def plot_contour(
    func, x_range, y_range, title, optimal_point, levels=None, backend="matplotlib", save_figs=False
):
    """
    Generic function to plot contour visualization with selectable backend

    Args:
        func: function to plot
        x_range: tuple of (min, max) for x axis
        y_range: tuple of (min, max) for y axis
        title: title for the plot
        optimal_point: tuple of (x, y) optimal coordinates
        levels: contour levels, if None uses default levels
        backend: 'matplotlib' or 'plotly'
        save_figs: whether to save figures to figs folder
    """
    if backend.lower() == "plotly":
        plot_contour_plotly(func, x_range, y_range, title, optimal_point, levels, save_figs)
    elif backend.lower() == "matplotlib":
        plot_contour_matplotlib(func, x_range, y_range, title, optimal_point, levels, save_figs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'.")


def plot_detailed_rosenbrock_contour():
    """
    Plot detailed contour visualization specifically for Rosenbrock function
    with fine-grained levels to show the banana-shaped valley clearly
    """
    from optimization_functions import rosenbrock_function

    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_function(X, Y)

    plt.figure(figsize=(8, 6))

    # Fine-grained contour lines for better banana shape visualization
    levels = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    contour_filled = plt.contourf(X, Y, Z, levels=levels, cmap="viridis", alpha=0.8)
    contour_lines = plt.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.5)

    plt.clabel(contour_lines, inline=True, fontsize=8, fmt="%g")
    plt.colorbar(contour_filled, label="f(x, y)")

    # Optimal solution point
    plt.plot(1, 1, "r*", markersize=20, label="Optimal (1,1)")

    # Emphasize banana-shaped valley
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Rosenbrock Function (Banana Function) - Detailed Contour", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
