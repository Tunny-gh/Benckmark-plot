"""
Main module for multi-objective optimization benchmark functions visualization.

This program visualizes 4 common multi-objective optimization benchmark functions
showing their Pareto fronts and objective space distributions.
"""

from multi_objective.mo_optimization_functions import MO_FUNCTIONS
from multi_objective.mo_visualization import plot_mo_function


def plot_all_mo_functions(backend="matplotlib", save_figs=False):
    """
    Plot all multi-objective optimization benchmark functions

    Args:
        backend: 'matplotlib' or 'plotly' for plotting backend
        save_figs: whether to save figures to figs folder
    """
    for func_name, func_info in MO_FUNCTIONS.items():
        print(f"Plotting {func_info['name']} using {backend}...")

        plot_mo_function(
            func_name=func_name,
            func_info=func_info,
            backend=backend,
            save_figs=save_figs,
        )


def main(backend="plotly", save_figs=False):
    """
    Main function

    Args:
        backend: 'matplotlib' or 'plotly' for plotting backend
        save_figs: whether to save figures to figs folder
    """
    print("=== Multi-Objective Optimization Benchmark Functions Visualization ===")
    print(
        "This program plots 4 common multi-objective optimization benchmark functions:"
    )
    print("1. DTLZ1 - Linear Pareto front")
    print("2. DTLZ2 - Concave Pareto front (circular)")
    print("3. ZDT1 - Convex Pareto front")
    print("4. ZDT2 - Non-convex Pareto front")
    print()
    print(f"Using {backend} backend for visualization.")
    print("Each function will be displayed with:")
    print("- Objective space plot showing sample points")
    print("- True Pareto front overlay (where known)")
    print("- Zoomed view of Pareto front region")
    print()

    # Plot all functions using the specified backend
    plot_all_mo_functions(backend=backend, save_figs=save_figs)

    print(
        "All multi-objective optimization function graphs have been plotted successfully!"
    )
    print(
        f"Each function shows objective space distribution and Pareto fronts using {backend}."
    )


if __name__ == "__main__":
    import sys

    # Parse command line arguments for backend selection and save option
    backend = "matplotlib"  # default
    save_figs = False  # default

    for arg in sys.argv[1:]:
        arg_lower = arg.lower()
        if arg_lower in ["matplotlib", "plotly"]:
            backend = arg_lower
        elif arg_lower in ["--save", "-s"]:
            save_figs = True
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python mo_main.py [matplotlib|plotly] [--save|-s]")
            print("Available backends: matplotlib, plotly")
            print("Use --save or -s to save figures to figs folder")
            sys.exit(1)

    if save_figs:
        print("Figures will be saved to 'figs' folder")

    main(backend=backend, save_figs=save_figs)
