"""
Optimization Functions Visualization Package

This package provides visualization tools for common optimization benchmark functions.
It includes implementations of 6 standard optimization functions and tools to visualize
them using 3D surface plots and contour plots.

Modules:
    optimization_functions: Contains implementations of optimization benchmark functions
    visualization: Contains plotting and visualization utilities
    main: Main module with the application entry point

Example:
    python main.py
"""

__version__ = "1.0.0"
__author__ = "Graph Visualization Project"

# Import main components for easy access
from .optimization_functions import FUNCTIONS
from .visualization import plot_function_3d, plot_contour

__all__ = ["FUNCTIONS", "plot_function_3d", "plot_contour"]
