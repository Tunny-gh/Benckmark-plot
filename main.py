"""
Main module for optimization benchmark functions visualization.

This program visualizes 6 common optimization benchmark functions using
3D surface plots and contour plots.
"""

import numpy as np
from optimization_functions import FUNCTIONS
from visualization import plot_function_3d


def plot_all_functions(backend="matplotlib"):
    """
    Plot all optimization benchmark functions using the function registry

    Args:
        backend: 'matplotlib' or 'plotly' for plotting backend
    """
    for func_name, func_info in FUNCTIONS.items():
        print(f"Plotting {func_info['name']} using {backend}...")

        # Special handling for Rosenbrock function to use log scale levels
        levels = func_info["levels"]
        if func_name == "rosenbrock":
            levels = np.logspace(0, 3, 20)

        plot_function_3d(
            func=func_info["function"],
            x_range=func_info["range"][0],
            y_range=func_info["range"][1],
            title=func_info["name"],
            optimal_point=func_info["optimal"],
            levels=levels,
            backend=backend,
        )


def main(backend="plotly"):
    """
    Main function

    Args:
        backend: 'matplotlib' or 'plotly' for plotting backend
    """
    print("=== Optimization Benchmark Functions Visualization ===")
    print("This program plots 6 common optimization benchmark functions:")
    print("1. Rosenbrock (Banana) - Global min: (1, 1)")
    print("2. Ackley - Global min: (0, 0)")
    print("3. Rastrigin - Global min: (0, 0)")
    print("4. Schwefel - Global min: (420.9687, 420.9687)")
    print("5. Sphere - Global min: (0, 0)")
    print("6. Griewank - Global min: (0, 0)")
    print()
    print(f"Using {backend} backend for visualization.")
    print("Each function will be displayed with 4 different 3D views:")
    print("- Top View (looking down)")
    print("- Front View (looking from front)")
    print("- Side View (looking from side)")
    print("- Perspective View (3D perspective)")
    print("Plus a separate contour plot for each function.")
    print()

    # Plot all functions using the specified backend
    plot_all_functions(backend=backend)

    print("All optimization function graphs have been plotted successfully!")
    print(
        f"Each function shows 4 different 3D views and a contour plot with optimal points marked in red using {backend}."
    )


if __name__ == "__main__":
    import sys

    # Parse command line arguments for backend selection
    backend = "matplotlib"  # default
    if len(sys.argv) > 1:
        backend_arg = sys.argv[1].lower()
        if backend_arg in ["matplotlib", "plotly"]:
            backend = backend_arg
        else:
            print(f"Unknown backend: {sys.argv[1]}. Using default 'matplotlib'.")
            print("Available backends: matplotlib, plotly")

    main(backend=backend)
