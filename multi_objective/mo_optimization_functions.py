"""
Multi-objective optimization benchmark functions module.

This module contains implementations of common multi-objective optimization
benchmark functions used for testing and comparing multi-objective optimization algorithms.
"""

import numpy as np


def dtlz1(x):
    """
    DTLZ1 function (scalable multi-objective function)
    For 2 objectives with n variables (default n=7)

    Args:
        x: input vector of length n (default: 7 variables)

    Returns:
        tuple of (f1, f2) objective values
    """
    n = len(x)
    # First part: sum of (xi - 0.5)^2 - cos(20*pi*(xi - 0.5))
    g = 100 * (n - 1 + np.sum((x[1:] - 0.5) ** 2 - np.cos(20 * np.pi * (x[1:] - 0.5))))

    f1 = 0.5 * x[0] * (1 + g)
    f2 = 0.5 * (1 - x[0]) * (1 + g)

    return f1, f2


def dtlz2(x):
    """
    DTLZ2 function (scalable multi-objective function)
    For 2 objectives with n variables (default n=12)

    Args:
        x: input vector of length n (default: 12 variables)

    Returns:
        tuple of (f1, f2) objective values
    """
    g = np.sum((x[1:] - 0.5) ** 2)

    f1 = (1 + g) * np.cos(x[0] * np.pi / 2)
    f2 = (1 + g) * np.sin(x[0] * np.pi / 2)

    return f1, f2


def zdt1(x):
    """
    ZDT1 function (Zitzler-Deb-Thiele function 1)
    For 2 objectives with n variables (default n=30)

    Args:
        x: input vector of length n (default: 30 variables)

    Returns:
        tuple of (f1, f2) objective values
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h

    return f1, f2


def zdt2(x):
    """
    ZDT2 function (Zitzler-Deb-Thiele function 2)
    For 2 objectives with n variables (default n=30)

    Args:
        x: input vector of length n (default: 30 variables)

    Returns:
        tuple of (f1, f2) objective values
    """
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - (f1 / g) ** 2
    f2 = g * h

    return f1, f2


# Function registry for easy access
MO_FUNCTIONS = {
    "dtlz1": {
        "function": dtlz1,
        "name": "DTLZ1 Function",
        "n_vars": 7,
        "bounds": [(0, 1)] * 7,
        "description": "DTLZ1 with linear Pareto front",
    },
    "dtlz2": {
        "function": dtlz2,
        "name": "DTLZ2 Function",
        "n_vars": 12,
        "bounds": [(0, 1)] * 12,
        "description": "DTLZ2 with concave Pareto front",
    },
    "zdt1": {
        "function": zdt1,
        "name": "ZDT1 Function",
        "n_vars": 30,
        "bounds": [(0, 1)] * 30,
        "description": "ZDT1 with convex Pareto front",
    },
    "zdt2": {
        "function": zdt2,
        "name": "ZDT2 Function",
        "n_vars": 30,
        "bounds": [(0, 1)] * 30,
        "description": "ZDT2 with non-convex Pareto front",
    },
}
