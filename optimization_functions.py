"""
Optimization benchmark functions module.

This module contains implementations of common optimization benchmark functions
used for testing and comparing optimization algorithms.
"""

import numpy as np


def rosenbrock_function(x, y, a=1, b=100):
    """
    Rosenbrock function (Banana function)
    f(x, y) = (a - x)^2 + b(y - x^2)^2
    Global minimum: (1, 1) with f(1, 1) = 0

    Args:
        x, y: input variables
        a: parameter (default: 1)
        b: parameter (default: 100)

    Returns:
        function value
    """
    return (a - x) ** 2 + b * (y - x**2) ** 2


def ackley_function(x, y, a=20, b=0.2, c=2 * np.pi):
    """
    Ackley function
    f(x, y) = -a * exp(-b * sqrt((x^2 + y^2)/2)) - exp((cos(c*x) + cos(c*y))/2) + a + e
    Global minimum: (0, 0) with f(0, 0) = 0

    Args:
        x, y: input variables
        a: parameter (default: 20)
        b: parameter (default: 0.2)
        c: parameter (default: 2*pi)

    Returns:
        function value
    """
    term1 = -a * np.exp(-b * np.sqrt((x**2 + y**2) / 2))
    term2 = -np.exp((np.cos(c * x) + np.cos(c * y)) / 2)
    return term1 + term2 + a + np.e


def rastrigin_function(x, y, A=10):
    """
    Rastrigin function
    f(x, y) = A*n + sum(xi^2 - A*cos(2*pi*xi))
    Global minimum: (0, 0) with f(0, 0) = 0

    Args:
        x, y: input variables
        A: parameter (default: 10)

    Returns:
        function value
    """
    return (
        2 * A + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    )


def schwefel_function(x, y):
    """
    Schwefel function
    f(x, y) = 418.9829*n - sum(xi * sin(sqrt(abs(xi))))
    Global minimum: (420.9687, 420.9687) with f ≈ 0

    Args:
        x, y: input variables

    Returns:
        function value
    """
    return 418.9829 * 2 - (
        x * np.sin(np.sqrt(np.abs(x))) + y * np.sin(np.sqrt(np.abs(y)))
    )


def sphere_function(x, y):
    """
    Sphere function
    f(x, y) = x^2 + y^2
    Global minimum: (0, 0) with f(0, 0) = 0

    Args:
        x, y: input variables

    Returns:
        function value
    """
    return x**2 + y**2


def griewank_function(x, y):
    """
    Griewank function
    f(x, y) = 1 + (x^2 + y^2)/4000 - cos(x)*cos(y/sqrt(2))
    Global minimum: (0, 0) with f(0, 0) = 0

    Args:
        x, y: input variables

    Returns:
        function value
    """
    return 1 + (x**2 + y**2) / 4000 - np.cos(x) * np.cos(y / np.sqrt(2))


# Function registry for easy access
FUNCTIONS = {
    "rosenbrock": {
        "function": rosenbrock_function,
        "name": "Rosenbrock Function",
        "range": ((-2, 2), (-1, 3)),
        "optimal": (1, 1),
        "levels": None,  # Will use log scale in plotting
    },
    "ackley": {
        "function": ackley_function,
        "name": "Ackley Function",
        "range": ((-5, 5), (-5, 5)),
        "optimal": (0, 0),
        "levels": None,
    },
    "rastrigin": {
        "function": rastrigin_function,
        "name": "Rastrigin Function",
        "range": ((-5, 5), (-5, 5)),
        "optimal": (0, 0),
        "levels": None,
    },
    "schwefel": {
        "function": schwefel_function,
        "name": "Schwefel Function",
        "range": ((-500, 500), (-500, 500)),
        "optimal": (420.9687, 420.9687),
        "levels": np.linspace(0, 1000, 20),
    },
    "sphere": {
        "function": sphere_function,
        "name": "Sphere Function",
        "range": ((-5, 5), (-5, 5)),
        "optimal": (0, 0),
        "levels": None,
    },
    "griewank": {
        "function": griewank_function,
        "name": "Griewank Function",
        "range": ((-10, 10), (-10, 10)),
        "optimal": (0, 0),
        "levels": None,
    },
}
