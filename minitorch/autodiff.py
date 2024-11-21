from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol, List


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    vals_list = list(vals)

    # Compute f(x + epsilon)
    vals_plus = vals_list[:arg] + [vals_list[arg] + epsilon] + vals_list[arg + 1 :]
    f_plus = f(*vals_plus)

    # Compute f(x - epsilon)
    vals_minus = vals_list[:arg] + [vals_list[arg] - epsilon] + vals_list[arg + 1 :]
    f_minus = f(*vals_minus)

    # Compute the central difference
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a derivative."""
        ...

    @property
    def unique_id(self) -> int:
        """Unique identifier for each variable"""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node"""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get parent variables"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to get derivatives for parent variables"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.

    order: List[Variable] = []

    seen = set()

    def visit(var: Variable) -> None:
        if var.unique_id in seen or var.is_constant():
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    visit(m)
        seen.add(var.unique_id)
        order.insert(0, var)

    visit(variable)
    return order

    # My previous approach, deprecated to change to answers for thoroughness
    # visited = set()
    # result = []

    # def dfs(var: Variable) -> None:
    #     if var.unique_id in visited or var.is_constant():
    #         return
    #     visited.add(var.unique_id)
    #     for parent in var.parents:
    #         dfs(parent)
    #     result.append(var)

    # dfs(variable)
    # return reversed(result)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # Initialize a dictionary to store derivatives for each variable
    derivatives = {}

    # Perform topological sort
    sorted_variables = topological_sort(variable)
    derivatives[variable.unique_id] = deriv
    # Iterate through the sorted variables
    for var in sorted_variables:
        # Get the current derivative for this variable
        d_output = derivatives[var.unique_id]

        # If it's a leaf node, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            # Apply the chain rule to get derivatives for parent variables
            for parent, grad in var.chain_rule(d_output):
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = grad
                else:
                    derivatives[parent.unique_id] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the saved tensors"""
        return self.saved_values
