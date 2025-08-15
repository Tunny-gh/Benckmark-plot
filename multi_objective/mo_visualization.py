"""
Multi-objective visualization module.

This module contains functions for plotting and visualizing multi-objective
optimization benchmark functions, including Pareto fronts and objective space plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def generate_pareto_front_samples(func, func_info, n_samples=1000):
    """
    Generate sample points and their objective values for visualization

    Args:
        func: multi-objective function
        func_info: function information dictionary
        n_samples: number of random samples to generate

    Returns:
        tuple of (samples, f1_values, f2_values)
    """
    n_vars = func_info["n_vars"]
    bounds = func_info["bounds"]

    # Generate random samples within bounds
    samples = []
    for i in range(n_samples):
        sample = []
        for j in range(n_vars):
            low, high = bounds[j]
            sample.append(np.random.uniform(low, high))
        samples.append(sample)

    # Evaluate objectives
    f1_values = []
    f2_values = []

    for sample in samples:
        f1, f2 = func(np.array(sample))
        f1_values.append(f1)
        f2_values.append(f2)

    return np.array(samples), np.array(f1_values), np.array(f2_values)


def generate_true_pareto_front(func_name, n_points=100):
    """
    Generate true Pareto front for known test functions

    Args:
        func_name: name of the function
        n_points: number of points on the Pareto front

    Returns:
        tuple of (f1_pareto, f2_pareto)
    """
    if func_name == "dtlz1":
        # Linear Pareto front: f1 + f2 = 0.5
        f1_pareto = np.linspace(0, 0.5, n_points)
        f2_pareto = 0.5 - f1_pareto

    elif func_name == "dtlz2":
        # Circular Pareto front: f1^2 + f2^2 = 1
        theta = np.linspace(0, np.pi / 2, n_points)
        f1_pareto = np.cos(theta)
        f2_pareto = np.sin(theta)

    elif func_name == "zdt1":
        # Convex Pareto front
        f1_pareto = np.linspace(0, 1, n_points)
        f2_pareto = 1 - np.sqrt(f1_pareto)

    elif func_name == "zdt2":
        # Non-convex Pareto front
        f1_pareto = np.linspace(0, 1, n_points)
        f2_pareto = 1 - f1_pareto**2

    else:
        # Default: return empty arrays
        f1_pareto = np.array([])
        f2_pareto = np.array([])

    return f1_pareto, f2_pareto


def plot_mo_function_matplotlib(func_name, func_info, save_figs=False):
    """
    Plot multi-objective function using matplotlib

    Args:
        func_name: name of the function
        func_info: function information dictionary
        save_figs: whether to save figures to figs folder
    """
    func = func_info["function"]

    # Generate sample points
    samples, f1_values, f2_values = generate_pareto_front_samples(
        func, func_info, n_samples=2000
    )

    # Generate true Pareto front
    f1_pareto, f2_pareto = generate_true_pareto_front(func_name)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Objective space with all samples
    ax1.scatter(
        f1_values, f2_values, alpha=0.6, s=20, c="lightblue", label="Random samples"
    )

    # Plot true Pareto front if available
    if len(f1_pareto) > 0:
        ax1.plot(f1_pareto, f2_pareto, "r-", linewidth=3, label="True Pareto Front")

    ax1.set_xlabel("f1 (Objective 1)", fontsize=12)
    ax1.set_ylabel("f2 (Objective 2)", fontsize=12)
    ax1.set_title(f"{func_info['name']} - Objective Space", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Zoomed view of Pareto front region
    if len(f1_pareto) > 0:
        # Filter points near Pareto front for better visualization
        if func_name in ["dtlz1", "dtlz2"]:
            # For DTLZ functions, focus on lower objective values
            mask = (f1_values <= 2) & (f2_values <= 2)
        else:
            # For ZDT functions, focus on the main front
            mask = (f1_values <= 1.2) & (f2_values <= 1.2)

        if np.any(mask):
            ax2.scatter(
                f1_values[mask],
                f2_values[mask],
                alpha=0.6,
                s=20,
                c="lightblue",
                label="Random samples",
            )

        ax2.plot(f1_pareto, f2_pareto, "r-", linewidth=3, label="True Pareto Front")
        ax2.set_xlabel("f1 (Objective 1)", fontsize=12)
        ax2.set_ylabel("f2 (Objective 2)", fontsize=12)
        ax2.set_title(f"{func_info['name']} - Pareto Front (Zoomed)", fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(
            0.5,
            0.5,
            "True Pareto front\nnot available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_title(f"{func_info['name']} - Pareto Front (Unknown)", fontsize=14)

    plt.tight_layout()

    # Save figure if requested
    if save_figs:
        os.makedirs("figs", exist_ok=True)
        clean_name = (
            func_info["name"].replace(" ", "_").replace("(", "").replace(")", "")
        )
        filename = f"figs/{clean_name}_MO.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved multi-objective plot: {filename}")

    plt.show()


def plot_mo_function_plotly(func_name, func_info, save_figs=False):
    """
    Plot multi-objective function using plotly

    Args:
        func_name: name of the function
        func_info: function information dictionary
        save_figs: whether to save figures to figs folder
    """
    func = func_info["function"]

    # Generate sample points
    samples, f1_values, f2_values = generate_pareto_front_samples(
        func, func_info, n_samples=2000
    )

    # Generate true Pareto front
    f1_pareto, f2_pareto = generate_true_pareto_front(func_name)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"{func_info['name']} - Objective Space",
            f"{func_info['name']} - Pareto Front",
        ),
        horizontal_spacing=0.1,
    )

    # Plot 1: All sample points
    fig.add_trace(
        go.Scatter(
            x=f1_values,
            y=f2_values,
            mode="markers",
            marker=dict(size=4, color="lightblue", opacity=0.6),
            name="Random samples",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Plot true Pareto front if available
    if len(f1_pareto) > 0:
        fig.add_trace(
            go.Scatter(
                x=f1_pareto,
                y=f2_pareto,
                mode="lines",
                line=dict(color="red", width=3),
                name="True Pareto Front",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Plot 2: Zoomed view
        if func_name in ["dtlz1", "dtlz2"]:
            mask = (f1_values <= 2) & (f2_values <= 2)
        else:
            mask = (f1_values <= 1.2) & (f2_values <= 1.2)

        if np.any(mask):
            fig.add_trace(
                go.Scatter(
                    x=f1_values[mask],
                    y=f2_values[mask],
                    mode="markers",
                    marker=dict(size=4, color="lightblue", opacity=0.6),
                    name="Random samples",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=f1_pareto,
                y=f2_pareto,
                mode="lines",
                line=dict(color="red", width=3),
                name="True Pareto Front",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Update layout
    fig.update_xaxes(title_text="f1 (Objective 1)", row=1, col=1)
    fig.update_yaxes(title_text="f2 (Objective 2)", row=1, col=1)
    fig.update_xaxes(title_text="f1 (Objective 1)", row=1, col=2)
    fig.update_yaxes(title_text="f2 (Objective 2)", row=1, col=2)

    fig.update_layout(
        title=f"{func_info['name']} - Multi-objective Visualization",
        height=500,
        width=1000,
        showlegend=True,
    )

    # Save figure if requested
    if save_figs:
        os.makedirs("figs", exist_ok=True)
        clean_name = (
            func_info["name"].replace(" ", "_").replace("(", "").replace(")", "")
        )
        filename = f"figs/{clean_name}_MO.html"
        fig.write_html(filename)
        print(f"Saved multi-objective plot: {filename}")

    fig.show()


def plot_mo_function(func_name, func_info, backend="matplotlib", save_figs=False):
    """
    Generic function to plot multi-objective visualization with selectable backend

    Args:
        func_name: name of the function
        func_info: function information dictionary
        backend: 'matplotlib' or 'plotly'
        save_figs: whether to save figures to figs folder
    """
    if backend.lower() == "plotly":
        plot_mo_function_plotly(func_name, func_info, save_figs)
    elif backend.lower() == "matplotlib":
        plot_mo_function_matplotlib(func_name, func_info, save_figs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'matplotlib' or 'plotly'.")
