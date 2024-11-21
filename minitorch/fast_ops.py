from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Decorator to compile a function for Numba."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        is_stride_aligned = True
        for i in range(len(out_shape)):
            if out_strides[i] != in_strides[i] or out_shape[i] != in_shape[i]:
                is_stride_aligned = False
                break

        # Fast path: stride-aligned case
        if is_stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])
            return

        # Slow path: handle broadcasting and different strides
        for i in prange(len(out)):
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)
            in_index = np.zeros(MAX_DIMS, dtype=np.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        is_stride_aligned = True
        for i in range(len(out_shape)):
            if (
                out_strides[i] != a_strides[i]
                or out_strides[i] != b_strides[i]
                or out_shape[i] != a_shape[i]
                or out_shape[i] != b_shape[i]
            ):
                is_stride_aligned = False
                break

        # Fast path: stride-aligned case (direct indexing)
        if is_stride_aligned:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])
        else:
            # Slow path: requires full index mapping and broadcasting
            # Each thread gets its own index arrays
            for i in prange(len(out)):
                # Create thread-local index arrays
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)
                b_index = np.zeros(MAX_DIMS, dtype=np.int32)

                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)

                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)

                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)

                out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # Initialize index arrays
        # Initialize index arrays
        reduce_size = a_shape[reduce_dim]

        # Parallel loop over output elements
        for i in prange(len(out)):
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)
            # Setup for this output position
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)

            # Pre-calculate the stride for reduce_dim
            reduce_stride = a_strides[reduce_dim]

            # Get initial position in a_storage
            out_index[reduce_dim] = 0
            base = index_to_position(out_index, a_strides)

            # Clean inner loop with no function calls or non-local writes
            temp = out[o]
            for s in range(reduce_size):
                # Just increment by stride instead of recalculating position
                curr_pos = base + s * reduce_stride
                temp = float(fn(float(temp), float(a_storage[curr_pos])))
            out[o] = temp

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    # Get relevant dimensions
    batch_size = out_shape[0]  # Can be 1 for non-batched
    M = a_shape[-2]  # Rows in output
    N = b_shape[-1]  # Columns in output
    K = a_shape[-1]  # Must equal b_shape[-2] (inner dimension)

    # Strides for the last two dimensions
    # a_inner_stride = a_strides[-1]
    # a_outer_stride = a_strides[-2]
    # b_inner_stride = b_strides[-2]
    # b_outer_stride = b_strides[-1]
    # out_inner_stride = out_strides[-1]
    # out_outer_stride = out_strides[-2]

    for batch in prange(batch_size):
        a_batch = batch * a_batch_stride
        b_batch = batch * b_batch_stride
        out_batch = batch * out_strides[0]

        for i in range(M):
            for j in range(N):
                # Initialize accumulator
                acc = 0.0

                # Inner loop - only multiplication, accumulate locally
                for k in range(K):
                    a_val = a_storage[a_batch + i * a_strides[-2] + k * a_strides[-1]]
                    b_val = b_storage[b_batch + k * b_strides[-2] + j * b_strides[-1]]
                    acc += a_val * b_val

                # Single write to out after accumulation
                out_pos = out_batch + i * out_strides[-2] + j * out_strides[-1]
                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
