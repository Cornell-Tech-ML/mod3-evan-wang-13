# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.1 and 3.2 Diagnostics
<details>
<summary>Diagnostics</summary>
        
```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/content/mod3-evan-wang-13/minitorch/fast_ops.py (164)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /content/mod3-evan-wang-13/minitorch/fast_ops.py (164) 
----------------------------------------------------------------------------------|loop #ID
    def _map(                                                                     | 
        out: Storage,                                                             | 
        out_shape: Shape,                                                         | 
        out_strides: Strides,                                                     | 
        in_storage: Storage,                                                      | 
        in_shape: Shape,                                                          | 
        in_strides: Strides,                                                      | 
    ) -> None:                                                                    | 
        # TODO: Implement for Task 3.1.                                           | 
        is_stride_aligned = True                                                  | 
        for i in range(len(out_shape)):                                           | 
            if out_strides[i] != in_strides[i] or out_shape[i] != in_shape[i]:    | 
                is_stride_aligned = False                                         | 
                break                                                             | 
                                                                                  | 
        # Fast path: stride-aligned case                                          | 
        if is_stride_aligned:                                                     | 
            for i in prange(len(out)):--------------------------------------------| #2
                out[i] = fn(in_storage[i])                                        | 
            return                                                                | 
                                                                                  | 
        # Slow path: handle broadcasting and different strides                    | 
        for i in prange(len(out)):------------------------------------------------| #3
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)------------------------| #0
            in_index = np.zeros(MAX_DIMS, dtype=np.int32)-------------------------| #1
            to_index(i, out_shape, out_index)                                     | 
            broadcast_index(out_index, out_shape, in_shape, in_index)             | 
            o = index_to_position(out_index, out_strides)                         | 
            j = index_to_position(in_index, in_strides)                           | 
            out[o] = fn(in_storage[j])                                            | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /content/mod3-evan-
wang-13/minitorch/fast_ops.py (187) is hoisted out of the parallel loop labelled
 #3 (it will be performed before the loop is executed and reused inside the 
loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /content/mod3-evan-
wang-13/minitorch/fast_ops.py (188) is hoisted out of the parallel loop labelled
 #3 (it will be performed before the loop is executed and reused inside the 
loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/content/mod3-evan-wang-13/minitorch/fast_ops.py (221)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /content/mod3-evan-wang-13/minitorch/fast_ops.py (221) 
---------------------------------------------------------------------------|loop #ID
    def _zip(                                                              | 
        out: Storage,                                                      | 
        out_shape: Shape,                                                  | 
        out_strides: Strides,                                              | 
        a_storage: Storage,                                                | 
        a_shape: Shape,                                                    | 
        a_strides: Strides,                                                | 
        b_storage: Storage,                                                | 
        b_shape: Shape,                                                    | 
        b_strides: Strides,                                                | 
    ) -> None:                                                             | 
        # TODO: Implement for Task 3.1.                                    | 
        is_stride_aligned = True                                           | 
        for i in range(len(out_shape)):                                    | 
            if (                                                           | 
                out_strides[i] != a_strides[i]                             | 
                or out_strides[i] != b_strides[i]                          | 
                or out_shape[i] != a_shape[i]                              | 
                or out_shape[i] != b_shape[i]                              | 
            ):                                                             | 
                is_stride_aligned = False                                  | 
                break                                                      | 
                                                                           | 
        # Fast path: stride-aligned case (direct indexing)                 | 
        if is_stride_aligned:                                              | 
            for i in prange(len(out)):-------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                    | 
        else:                                                              | 
            # Slow path: requires full index mapping and broadcasting      | 
            # Each thread gets its own index arrays                        | 
            for i in prange(len(out)):-------------------------------------| #8
                # Create thread-local index arrays                         | 
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)-------------| #4
                a_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------| #5
                b_index = np.zeros(MAX_DIMS, dtype=np.int32)---------------| #6
                                                                           | 
                to_index(i, out_shape, out_index)                          | 
                o = index_to_position(out_index, out_strides)              | 
                                                                           | 
                broadcast_index(out_index, out_shape, a_shape, a_index)    | 
                j = index_to_position(a_index, a_strides)                  | 
                                                                           | 
                broadcast_index(out_index, out_shape, b_shape, b_index)    | 
                k = index_to_position(b_index, b_strides)                  | 
                                                                           | 
                out[o] = fn(a_storage[j], b_storage[k])                    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
 
Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /content/mod3-evan-
wang-13/minitorch/fast_ops.py (253) is hoisted out of the parallel loop labelled
 #8 (it will be performed before the loop is executed and reused inside the 
loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /content/mod3-evan-
wang-13/minitorch/fast_ops.py (254) is hoisted out of the parallel loop labelled
 #8 (it will be performed before the loop is executed and reused inside the 
loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at /content/mod3-evan-
wang-13/minitorch/fast_ops.py (255) is hoisted out of the parallel loop labelled
 #8 (it will be performed before the loop is executed and reused inside the 
loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/content/mod3-evan-wang-13/minitorch/fast_ops.py (292)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /content/mod3-evan-wang-13/minitorch/fast_ops.py (292) 
--------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                | 
        out: Storage,                                                           | 
        out_shape: Shape,                                                       | 
        out_strides: Strides,                                                   | 
        a_storage: Storage,                                                     | 
        a_shape: Shape,                                                         | 
        a_strides: Strides,                                                     | 
        reduce_dim: int,                                                        | 
    ) -> None:                                                                  | 
        # TODO: Implement for Task 3.1.                                         | 
        # Initialize index arrays                                               | 
        # Initialize index arrays                                               | 
        reduce_size = a_shape[reduce_dim]                                       | 
                                                                                | 
        # Parallel loop over output elements                                    | 
        for i in prange(len(out)):----------------------------------------------| #10
            out_index = np.zeros(MAX_DIMS, dtype=np.int32)----------------------| #9
            # Setup for this output position                                    | 
            to_index(i, out_shape, out_index)                                   | 
            o = index_to_position(out_index, out_strides)                       | 
                                                                                | 
            # Pre-calculate the stride for reduce_dim                           | 
            reduce_stride = a_strides[reduce_dim]                               | 
                                                                                | 
            # Get initial position in a_storage                                 | 
            out_index[reduce_dim] = 0                                           | 
            base = index_to_position(out_index, a_strides)                      | 
                                                                                | 
            # Clean inner loop with no function calls or non-local writes       | 
            temp = out[o]                                                       | 
            for s in range(reduce_size):                                        | 
                # Just increment by stride instead of recalculating position    | 
                curr_pos = base + s * reduce_stride                             | 
                temp = float(fn(float(temp), float(a_storage[curr_pos])))       | 
            out[o] = temp                                                       | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #10, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--10 is a parallel loop
   +--9 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--9 (serial)


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as 
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at /content/mod3-evan-
wang-13/minitorch/fast_ops.py (308) is hoisted out of the parallel loop labelled
 #10 (it will be performed before the loop is executed and reused inside the 
loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/content/mod3-evan-wang-13/minitorch/fast_ops.py (331)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /content/mod3-evan-wang-13/minitorch/fast_ops.py (331) 
------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                              | 
    out: Storage,                                                                         | 
    out_shape: Shape,                                                                     | 
    out_strides: Strides,                                                                 | 
    a_storage: Storage,                                                                   | 
    a_shape: Shape,                                                                       | 
    a_strides: Strides,                                                                   | 
    b_storage: Storage,                                                                   | 
    b_shape: Shape,                                                                       | 
    b_strides: Strides,                                                                   | 
) -> None:                                                                                | 
    """NUMBA tensor matrix multiply function.                                             | 
                                                                                          | 
    Should work for any tensor shapes that broadcast as long as                           | 
                                                                                          | 
    ```                                                                                   | 
    assert a_shape[-1] == b_shape[-2]                                                     | 
    ```                                                                                   | 
                                                                                          | 
    Optimizations:                                                                        | 
                                                                                          | 
    * Outer loop in parallel                                                              | 
    * No index buffers or function calls                                                  | 
    * Inner loop should have no global writes, 1 multiply.                                | 
                                                                                          | 
                                                                                          | 
    Args:                                                                                 | 
    ----                                                                                  | 
        out (Storage): storage for `out` tensor                                           | 
        out_shape (Shape): shape for `out` tensor                                         | 
        out_strides (Strides): strides for `out` tensor                                   | 
        a_storage (Storage): storage for `a` tensor                                       | 
        a_shape (Shape): shape for `a` tensor                                             | 
        a_strides (Strides): strides for `a` tensor                                       | 
        b_storage (Storage): storage for `b` tensor                                       | 
        b_shape (Shape): shape for `b` tensor                                             | 
        b_strides (Strides): strides for `b` tensor                                       | 
                                                                                          | 
    Returns:                                                                              | 
    -------                                                                               | 
        None : Fills in `out`                                                             | 
                                                                                          | 
    """                                                                                   | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                | 
                                                                                          | 
    # TODO: Implement for Task 3.2.                                                       | 
    # Get relevant dimensions                                                             | 
    batch_size = out_shape[0]  # Can be 1 for non-batched                                 | 
    M = a_shape[-2]  # Rows in output                                                     | 
    N = b_shape[-1]  # Columns in output                                                  | 
    K = a_shape[-1]  # Must equal b_shape[-2] (inner dimension)                           | 
                                                                                          | 
    # Strides for the last two dimensions                                                 | 
    # a_inner_stride = a_strides[-1]                                                      | 
    # a_outer_stride = a_strides[-2]                                                      | 
    # b_inner_stride = b_strides[-2]                                                      | 
    # b_outer_stride = b_strides[-1]                                                      | 
    # out_inner_stride = out_strides[-1]                                                  | 
    # out_outer_stride = out_strides[-2]                                                  | 
                                                                                          | 
    for batch in prange(batch_size):------------------------------------------------------| #11
        a_batch = batch * a_batch_stride                                                  | 
        b_batch = batch * b_batch_stride                                                  | 
        out_batch = batch * out_strides[0]                                                | 
                                                                                          | 
        for i in range(M):                                                                | 
            for j in range(N):                                                            | 
                # Initialize accumulator                                                  | 
                acc = 0.0                                                                 | 
                                                                                          | 
                # Inner loop - only multiplication, accumulate locally                    | 
                for k in range(K):                                                        | 
                    a_val = a_storage[a_batch + i * a_strides[-2] + k * a_strides[-1]]    | 
                    b_val = b_storage[b_batch + k * b_strides[-2] + j * b_strides[-1]]    | 
                    acc += a_val * b_val                                                  | 
                                                                                          | 
                # Single write to out after accumulation                                  | 
                out_pos = out_batch + i * out_strides[-2] + j * out_strides[-1]           | 
                out[out_pos] = acc                                                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #11).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

</details>

# 3.4: Comparison Graph
- The values were calculated using the helper 'timing.py' file provided
Specific values outputted by script:
```
Timing summary
Size: 64
    fast: 0.00300
    gpu: 0.00566
Size: 128
    fast: 0.01436
    gpu: 0.01242
Size: 256
    fast: 0.09176
    gpu: 0.04835
Size: 512
    fast: 1.23617
    gpu: 0.26183
Size: 1024
    fast: 12.62239
    gpu: 0.90193
```
![runtime-comparison](https://github.com/user-attachments/assets/e73b6fa3-6b9c-45b0-8f50-3e93a0e14f6a)


# 3.5
Small Model
<details>
<summary>GPU on Split Data Training Logs Dropdown</summary>

```Epoch   0 | loss   6.5091 | correct  32 | time  3.4345s
Epoch  10 | loss   6.0863 | correct  42 | time  1.9132s
Epoch  20 | loss   5.2233 | correct  47 | time  1.3070s
Epoch  30 | loss   4.9386 | correct  48 | time  1.3130s
Epoch  40 | loss   3.7960 | correct  48 | time  1.3085s
Epoch  50 | loss   2.7895 | correct  48 | time  1.3177s
Epoch  60 | loss   2.3967 | correct  49 | time  1.3129s
Epoch  70 | loss   1.8395 | correct  47 | time  1.7589s
Epoch  80 | loss   1.1130 | correct  50 | time  1.3903s
Epoch  90 | loss   1.4565 | correct  48 | time  1.3039s
Epoch 100 | loss   1.3003 | correct  48 | time  1.4061s
Epoch 110 | loss   0.7179 | correct  48 | time  1.3801s
Epoch 120 | loss   0.7365 | correct  49 | time  1.3648s
Epoch 130 | loss   1.3216 | correct  50 | time  1.8494s
Epoch 140 | loss   0.5193 | correct  50 | time  1.3963s
Epoch 150 | loss   0.7683 | correct  50 | time  1.3188s
Epoch 160 | loss   0.5869 | correct  49 | time  1.3142s
Epoch 170 | loss   0.7209 | correct  50 | time  1.3182s
Epoch 180 | loss   0.7176 | correct  50 | time  1.3184s
Epoch 190 | loss   1.3205 | correct  50 | time  1.3636s
Epoch 200 | loss   0.2897 | correct  49 | time  1.8674s
Epoch 210 | loss   1.9033 | correct  49 | time  1.3072s
Epoch 220 | loss   0.6111 | correct  50 | time  1.3549s
Epoch 230 | loss   0.0865 | correct  50 | time  1.3142s
Epoch 240 | loss   1.3652 | correct  49 | time  1.3179s
Epoch 250 | loss   0.4097 | correct  49 | time  1.3043s
Epoch 260 | loss   0.9367 | correct  50 | time  1.8157s
Epoch 270 | loss   0.7216 | correct  49 | time  1.4429s
Epoch 280 | loss   0.0957 | correct  50 | time  1.3684s
Epoch 290 | loss   0.0817 | correct  50 | time  1.3085s
Epoch 300 | loss   0.6163 | correct  50 | time  1.3032s
Epoch 310 | loss   0.2272 | correct  50 | time  1.3092s
Epoch 320 | loss   0.2445 | correct  50 | time  1.7842s
Epoch 330 | loss   0.1378 | correct  50 | time  1.3267s
Epoch 340 | loss   0.1353 | correct  50 | time  1.3149s
Epoch 350 | loss   0.1132 | correct  50 | time  1.2983s
Epoch 360 | loss   0.5835 | correct  50 | time  1.3391s
Epoch 370 | loss   0.0843 | correct  50 | time  1.3153s
Epoch 380 | loss   0.6225 | correct  50 | time  1.3056s
Epoch 390 | loss   0.6490 | correct  50 | time  1.9018s
Epoch 400 | loss   0.3157 | correct  50 | time  1.3578s
Epoch 410 | loss   0.0915 | correct  50 | time  1.3760s
Epoch 420 | loss   0.7524 | correct  49 | time  1.3565s
Epoch 430 | loss   1.0445 | correct  49 | time  1.3843s
Epoch 440 | loss   0.1162 | correct  50 | time  1.9050s
Epoch 450 | loss   0.3530 | correct  50 | time  1.2955s
Epoch 460 | loss   0.1337 | correct  50 | time  1.3130s
Epoch 470 | loss   0.3193 | correct  50 | time  1.3007s
Epoch 480 | loss   0.5998 | correct  50 | time  1.2960s
Epoch 490 | loss   0.0614 | correct  50 | time  1.3075s
Epoch 500 | loss   0.1024 | correct  50 | time  1.4292s

Average epoch time: 1.4292s
```
</details>

<details>
<summary>CPU on Split Data Training Logs Dropdown</summary>
</details>details>

<details>
<summary>GPU on XOR Data Training Logs Dropdown</summary>
</details>details>

<details>
<summary>CPU on XOR Data Training Logs Dropdown</summary>
</details>details>

<details>
<summary>GPU on Simple Data Training Logs Dropdown</summary>
</details>details>

<details>
<summary>CPU on Simple Data Training Logs Dropdown</summary>
</details>details>
