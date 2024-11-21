"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


def id(x: float) -> float:
    """Identity function."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers."""
    return x + y


def neg(x: float) -> float:
    """Negate a number."""
    return -x


def lt(x: float, y: float) -> bool:
    """Check if x is less than y."""
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if x is equal to y."""
    return x == y


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close (within 1e-2)."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function."""
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Compute the natural logarithm."""
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Compute the gradient of log."""
    return d / x


def inv(x: float) -> float:
    """Compute the inverse of x."""
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """Compute the gradient of inv."""
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the gradient of ReLU."""
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        fn: A function that takes a float and returns a float.

    Returns:
    -------
        A function that takes an iterable of floats and returns an iterable of floats
        with the input function applied to each element.

    """

    def inner(ls: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in ls]

    return inner


def zipWith(
    fn: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]
) -> Iterable[float]:
    """Combine elements from two iterables using a given function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        iter1: An iterable of floats.
        iter2: An iterable of floats.

    Returns:
    -------
        An iterable of floats with the function applied to each pair of elements.

    """
    return [fn(x, y) for x, y in zip(iter1, iter2)]


def reduce(
    fn: Callable[[float, float], float], iter: Iterable[float], initial: float
) -> float:
    """Reduce an iterable to a single value using a given function.

    Args:
    ----
        fn: A function that takes two floats and returns a float.
        iter: An iterable of floats.
        initial: A float.

    Returns:
    -------
        A float constructed from the function applied to each element sequentially.

    """
    result = initial
    for x in iter:
        result = fn(result, x)
    return result


def negList(iter: Iterable[float]) -> Iterable[float]:
    """Negate a list.

    Args:
    ----
        iter: An iterable of floats.

    Returns:
    -------
        An iterable of floats with each element negated.

    """
    return map(neg)(iter)


def addLists(iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """Add two lists.

    Args:
    ----
        iter1: An iterable of floats.
        iter2: An iterable of floats.

    Returns:
    -------
        An iterable of floats with corresponding elements from input lists added.

    """
    return zipWith(add, iter1, iter2)


def sum(iter: Iterable[float]) -> float:
    """Sum all elements in a list using reduce.

    Args:
    ----
        iter: An iterable of floats.

    Returns:
    -------
        A float that is the sum of all elements in the list.

    """
    return reduce(add, iter, 0.0)


def prod(iter: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce.

    Args:
    ----
        iter: An iterable of floats.

    Returns:
    -------
        A float that is the product of all elements in the list.

    """
    return reduce(mul, iter, 1.0)
