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
### Small Model (100 hidden layer size)
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
        
```
Epoch   0 | loss   8.3173 | correct  24 | time 14.3743s
Epoch  10 | loss   4.8828 | correct  42 | time  0.4910s
Epoch  20 | loss   3.9444 | correct  45 | time  0.6731s
Epoch  30 | loss   4.0222 | correct  46 | time  0.4936s
Epoch  40 | loss   4.6975 | correct  46 | time  0.9031s
Epoch  50 | loss   1.9820 | correct  46 | time  0.5152s
Epoch  60 | loss   1.9748 | correct  47 | time  0.4895s
Epoch  70 | loss   3.6120 | correct  46 | time  0.5038s
Epoch  80 | loss   1.9646 | correct  48 | time  0.5040s
Epoch  90 | loss   2.1655 | correct  48 | time  0.5070s
Epoch 100 | loss   1.3377 | correct  47 | time  0.5040s
Epoch 110 | loss   0.5948 | correct  49 | time  0.4917s
Epoch 120 | loss   1.3632 | correct  48 | time  0.5222s
Epoch 130 | loss   1.2109 | correct  49 | time  0.5038s
Epoch 140 | loss   1.2031 | correct  48 | time  0.5049s
Epoch 150 | loss   1.1489 | correct  50 | time  0.5017s
Epoch 160 | loss   0.4585 | correct  50 | time  0.5020s
Epoch 170 | loss   1.1690 | correct  48 | time  0.6301s
Epoch 180 | loss   0.1176 | correct  49 | time  0.5005s
Epoch 190 | loss   0.5043 | correct  50 | time  0.9541s
Epoch 200 | loss   0.2359 | correct  49 | time  0.5007s
Epoch 210 | loss   1.5508 | correct  50 | time  0.4996s
Epoch 220 | loss   0.3888 | correct  49 | time  0.5078s
Epoch 230 | loss   1.4773 | correct  50 | time  0.5171s
Epoch 240 | loss   0.1010 | correct  49 | time  0.5091s
Epoch 250 | loss   0.4153 | correct  49 | time  0.4980s
Epoch 260 | loss   0.2957 | correct  50 | time  0.5030s
Epoch 270 | loss   0.0506 | correct  50 | time  0.5044s
Epoch 280 | loss   0.3320 | correct  49 | time  0.9080s
Epoch 290 | loss   0.9676 | correct  50 | time  0.5242s
Epoch 300 | loss   0.8026 | correct  49 | time  0.6633s
Epoch 310 | loss   1.2791 | correct  49 | time  0.5039s
Epoch 320 | loss   1.0232 | correct  48 | time  0.5096s
Epoch 330 | loss   1.5580 | correct  50 | time  0.5135s
Epoch 340 | loss   0.2419 | correct  50 | time  0.5053s
Epoch 350 | loss   0.1273 | correct  49 | time  0.5072s
Epoch 360 | loss   0.8828 | correct  50 | time  0.5017s
Epoch 370 | loss   0.8990 | correct  50 | time  0.5012s
Epoch 380 | loss   1.5775 | correct  49 | time  0.5039s
Epoch 390 | loss   0.1953 | correct  50 | time  0.9089s
Epoch 400 | loss   0.9420 | correct  50 | time  0.5014s
Epoch 410 | loss   0.6194 | correct  50 | time  0.5034s
Epoch 420 | loss   0.1077 | correct  50 | time  0.4988s
Epoch 430 | loss   0.0929 | correct  50 | time  0.4995s
Epoch 440 | loss   0.4083 | correct  50 | time  0.5062s
Epoch 450 | loss   0.2095 | correct  50 | time  0.5108s
Epoch 460 | loss   0.5673 | correct  50 | time  0.5017s
Epoch 470 | loss   0.6173 | correct  50 | time  0.4974s
Epoch 480 | loss   0.4800 | correct  50 | time  0.5021s
Epoch 490 | loss   0.0274 | correct  50 | time  0.5231s
Epoch 500 | loss   0.2453 | correct  50 | time  0.5864s

Average epoch time: 0.5864s
```
</details>

<details>
<summary>GPU on XOR Data Training Logs Dropdown</summary>
        
```
Epoch   0 | loss   7.0592 | correct  32 | time  3.9925s
Epoch  10 | loss   4.7726 | correct  43 | time  1.3164s
Epoch  20 | loss   5.9306 | correct  40 | time  1.3118s
Epoch  30 | loss   3.6014 | correct  45 | time  1.3165s
Epoch  40 | loss   3.0764 | correct  46 | time  1.5331s
Epoch  50 | loss   2.1698 | correct  47 | time  1.2999s
Epoch  60 | loss   2.1910 | correct  48 | time  1.9034s
Epoch  70 | loss   1.2753 | correct  45 | time  1.3154s
Epoch  80 | loss   2.3483 | correct  48 | time  1.3129s
Epoch  90 | loss   1.1134 | correct  47 | time  1.3207s
Epoch 100 | loss   2.3171 | correct  46 | time  1.3781s
Epoch 110 | loss   1.7772 | correct  47 | time  1.3762s
Epoch 120 | loss   1.4012 | correct  49 | time  1.9550s
Epoch 130 | loss   1.7757 | correct  48 | time  1.3756s
Epoch 140 | loss   1.6064 | correct  49 | time  1.3107s
Epoch 150 | loss   0.7710 | correct  48 | time  1.3589s
Epoch 160 | loss   1.0814 | correct  48 | time  1.3366s
Epoch 170 | loss   1.1791 | correct  48 | time  1.3117s
Epoch 180 | loss   2.2761 | correct  48 | time  1.5915s
Epoch 190 | loss   1.8973 | correct  48 | time  2.2769s
Epoch 200 | loss   1.6177 | correct  49 | time  1.3111s
Epoch 210 | loss   2.5411 | correct  47 | time  1.3072s
Epoch 220 | loss   1.5099 | correct  49 | time  1.3227s
Epoch 230 | loss   1.3098 | correct  50 | time  1.3050s
Epoch 240 | loss   1.6125 | correct  50 | time  1.3408s
Epoch 250 | loss   0.7542 | correct  49 | time  1.7890s
Epoch 260 | loss   0.3304 | correct  49 | time  1.4284s
Epoch 270 | loss   0.3799 | correct  49 | time  1.3600s
Epoch 280 | loss   1.0261 | correct  49 | time  1.3663s
Epoch 290 | loss   0.6770 | correct  48 | time  1.3052s
Epoch 300 | loss   1.1400 | correct  50 | time  1.3123s
Epoch 310 | loss   0.6126 | correct  48 | time  1.3364s
Epoch 320 | loss   1.0510 | correct  49 | time  1.8300s
Epoch 330 | loss   1.3222 | correct  48 | time  1.3094s
Epoch 340 | loss   0.5957 | correct  49 | time  1.2993s
Epoch 350 | loss   1.8540 | correct  49 | time  1.3037s
Epoch 360 | loss   1.3079 | correct  50 | time  1.3549s
Epoch 370 | loss   2.4347 | correct  50 | time  1.2990s
Epoch 380 | loss   0.8072 | correct  49 | time  1.2983s
Epoch 390 | loss   0.4972 | correct  50 | time  1.9259s
Epoch 400 | loss   1.8003 | correct  48 | time  1.2979s
Epoch 410 | loss   0.5574 | correct  49 | time  1.3581s
Epoch 420 | loss   0.3975 | correct  50 | time  1.3638s
Epoch 430 | loss   0.4363 | correct  50 | time  1.3847s
Epoch 440 | loss   1.1218 | correct  49 | time  1.2996s
Epoch 450 | loss   1.1645 | correct  50 | time  1.6064s
Epoch 460 | loss   0.5753 | correct  50 | time  1.6089s
Epoch 470 | loss   1.6002 | correct  48 | time  1.3196s
Epoch 480 | loss   1.0059 | correct  49 | time  1.3096s
Epoch 490 | loss   1.4528 | correct  50 | time  1.3331s
Epoch 500 | loss   1.2214 | correct  50 | time  1.4285s

Average epoch time: 1.4285s
```
</details>

<details>
<summary>CPU on XOR Data Training Logs Dropdown</summary>

```
Epoch   0 | loss   6.3792 | correct  30 | time 14.4099s
Epoch  10 | loss   6.1569 | correct  41 | time  0.8190s
Epoch  20 | loss   5.3271 | correct  40 | time  0.5041s
Epoch  30 | loss   4.0876 | correct  39 | time  0.7312s
Epoch  40 | loss   3.6035 | correct  44 | time  0.5029s
Epoch  50 | loss   3.3481 | correct  44 | time  0.5112s
Epoch  60 | loss   3.2935 | correct  40 | time  0.5030s
Epoch  70 | loss   1.9606 | correct  42 | time  0.5291s
Epoch  80 | loss   2.7872 | correct  42 | time  0.5064s
Epoch  90 | loss   3.9039 | correct  42 | time  0.5081s
Epoch 100 | loss   4.6003 | correct  42 | time  0.4998s
Epoch 110 | loss   1.7506 | correct  44 | time  0.5040s
Epoch 120 | loss   2.1685 | correct  44 | time  0.5035s
Epoch 130 | loss   3.6478 | correct  47 | time  0.5033s
Epoch 140 | loss   1.1007 | correct  47 | time  0.8948s
Epoch 150 | loss   3.1815 | correct  47 | time  0.5053s
Epoch 160 | loss   1.5001 | correct  48 | time  0.5010s
Epoch 170 | loss   1.6614 | correct  49 | time  0.5092s
Epoch 180 | loss   0.7751 | correct  49 | time  0.5020s
Epoch 190 | loss   2.2054 | correct  44 | time  0.5105s
Epoch 200 | loss   1.3886 | correct  49 | time  0.5054s
Epoch 210 | loss   1.1212 | correct  49 | time  0.5062s
Epoch 220 | loss   2.0953 | correct  50 | time  0.5097s
Epoch 230 | loss   1.0113 | correct  49 | time  0.5517s
Epoch 240 | loss   1.3356 | correct  49 | time  0.5088s
Epoch 250 | loss   0.5827 | correct  49 | time  0.9417s
Epoch 260 | loss   1.2013 | correct  50 | time  0.4980s
Epoch 270 | loss   1.0948 | correct  50 | time  0.4931s
Epoch 280 | loss   1.7276 | correct  50 | time  0.5081s
Epoch 290 | loss   1.0174 | correct  49 | time  0.5306s
Epoch 300 | loss   1.4438 | correct  50 | time  0.5026s
Epoch 310 | loss   0.8212 | correct  50 | time  0.5044s
Epoch 320 | loss   0.9641 | correct  49 | time  0.5032s
Epoch 330 | loss   1.1254 | correct  50 | time  0.5052s
Epoch 340 | loss   0.6550 | correct  50 | time  0.8104s
Epoch 350 | loss   1.0036 | correct  50 | time  0.5165s
Epoch 360 | loss   0.3221 | correct  50 | time  0.8993s
Epoch 370 | loss   0.2023 | correct  50 | time  0.5076s
Epoch 380 | loss   1.0245 | correct  50 | time  0.5036s
Epoch 390 | loss   1.1645 | correct  50 | time  0.5085s
Epoch 400 | loss   0.8279 | correct  50 | time  0.5098s
Epoch 410 | loss   0.8174 | correct  50 | time  0.4987s
Epoch 420 | loss   0.3393 | correct  50 | time  0.5058s
Epoch 430 | loss   0.2066 | correct  50 | time  0.5042s
Epoch 440 | loss   0.2080 | correct  49 | time  0.5255s
Epoch 450 | loss   0.5329 | correct  50 | time  0.5067s
Epoch 460 | loss   0.6538 | correct  50 | time  0.5110s
Epoch 470 | loss   0.7925 | correct  50 | time  0.9201s
Epoch 480 | loss   0.3553 | correct  50 | time  0.5043s
Epoch 490 | loss   0.5589 | correct  50 | time  0.5416s
Epoch 500 | loss   0.8147 | correct  50 | time  0.5836s

Average epoch time: 0.5836s
```
</details>

<details>
<summary>GPU on Simple Data Training Logs Dropdown</summary>

```
Epoch   0 | loss   7.0046 | correct  42 | time  3.3206s
Epoch  10 | loss   2.4966 | correct  47 | time  1.8572s
Epoch  20 | loss   3.3400 | correct  44 | time  1.3182s
Epoch  30 | loss   1.3177 | correct  45 | time  1.3064s
Epoch  40 | loss   1.4627 | correct  45 | time  1.3171s
Epoch  50 | loss   0.9303 | correct  46 | time  1.3160s
Epoch  60 | loss   1.1671 | correct  49 | time  1.3405s
Epoch  70 | loss   1.1779 | correct  47 | time  1.4482s
Epoch  80 | loss   1.3923 | correct  47 | time  1.7935s
Epoch  90 | loss   0.9373 | correct  50 | time  1.3033s
Epoch 100 | loss   1.2771 | correct  50 | time  1.3601s
Epoch 110 | loss   0.5825 | correct  47 | time  1.3465s
Epoch 120 | loss   1.2861 | correct  45 | time  1.4647s
Epoch 130 | loss   1.1947 | correct  50 | time  1.3587s
Epoch 140 | loss   0.9337 | correct  50 | time  1.8874s
Epoch 150 | loss   0.9186 | correct  49 | time  1.3005s
Epoch 160 | loss   1.7911 | correct  49 | time  1.3223s
Epoch 170 | loss   1.4207 | correct  47 | time  1.3105s
Epoch 180 | loss   1.8398 | correct  48 | time  1.2864s
Epoch 190 | loss   0.6253 | correct  48 | time  1.3038s
Epoch 200 | loss   1.0167 | correct  50 | time  1.6292s
Epoch 210 | loss   0.3833 | correct  49 | time  1.5602s
Epoch 220 | loss   3.6469 | correct  50 | time  1.3148s
Epoch 230 | loss   0.6636 | correct  50 | time  1.3014s
Epoch 240 | loss   0.7673 | correct  50 | time  1.3034s
Epoch 250 | loss   1.4199 | correct  49 | time  1.3498s
Epoch 260 | loss   0.6887 | correct  50 | time  1.3699s
Epoch 270 | loss   0.5482 | correct  50 | time  1.6937s
Epoch 280 | loss   0.7509 | correct  50 | time  1.4050s
Epoch 290 | loss   0.7678 | correct  50 | time  1.2985s
Epoch 300 | loss   0.5053 | correct  50 | time  1.2997s
Epoch 310 | loss   0.3195 | correct  49 | time  1.3012s
Epoch 320 | loss   0.2148 | correct  50 | time  1.3131s
Epoch 330 | loss   0.3553 | correct  49 | time  1.9387s
Epoch 340 | loss   0.1963 | correct  50 | time  1.3004s
Epoch 350 | loss   0.2186 | correct  49 | time  1.3042s
Epoch 360 | loss   0.2484 | correct  49 | time  1.3321s
Epoch 370 | loss   0.6561 | correct  50 | time  1.3019s
Epoch 380 | loss   0.5914 | correct  50 | time  1.3107s
Epoch 390 | loss   1.5012 | correct  50 | time  1.4341s
Epoch 400 | loss   0.3734 | correct  50 | time  1.7853s
Epoch 410 | loss   0.9818 | correct  48 | time  1.3561s
Epoch 420 | loss   0.0760 | correct  50 | time  1.3592s
Epoch 430 | loss   0.2070 | correct  49 | time  1.3494s
Epoch 440 | loss   0.1165 | correct  50 | time  1.3194s
Epoch 450 | loss   0.2625 | correct  50 | time  1.3096s
Epoch 460 | loss   0.4943 | correct  50 | time  1.9453s
Epoch 470 | loss   0.6167 | correct  49 | time  1.3095s
Epoch 480 | loss   0.3627 | correct  50 | time  1.3101s
Epoch 490 | loss   0.1802 | correct  50 | time  1.2975s
Epoch 500 | loss   1.4009 | correct  48 | time  1.4224s

Average epoch time: 1.4224s
```
</details>

<details>
<summary>CPU on Simple Data Training Logs Dropdown</summary>

```
Epoch   0 | loss   4.9362 | correct  44 | time 14.2994s
Epoch  10 | loss   1.3063 | correct  48 | time  0.5125s
Epoch  20 | loss   1.0022 | correct  48 | time  0.5021s
Epoch  30 | loss   1.9730 | correct  49 | time  0.5051s
Epoch  40 | loss   0.8845 | correct  50 | time  0.8599s
Epoch  50 | loss   0.6089 | correct  48 | time  0.5362s
Epoch  60 | loss   0.8486 | correct  48 | time  0.5084s
Epoch  70 | loss   0.3767 | correct  50 | time  0.5029s
Epoch  80 | loss   0.4823 | correct  50 | time  0.5036s
Epoch  90 | loss   0.6848 | correct  50 | time  0.6538s
Epoch 100 | loss   0.7523 | correct  50 | time  0.4969s
Epoch 110 | loss   0.6311 | correct  50 | time  0.9604s
Epoch 120 | loss   0.2060 | correct  50 | time  0.5098s
Epoch 130 | loss   0.2112 | correct  50 | time  0.5020s
Epoch 140 | loss   0.1609 | correct  50 | time  0.5034s
Epoch 150 | loss   0.0332 | correct  50 | time  0.5065s
Epoch 160 | loss   0.1570 | correct  50 | time  0.5037s
Epoch 170 | loss   0.4806 | correct  50 | time  0.5051s
Epoch 180 | loss   0.6186 | correct  50 | time  0.5080s
Epoch 190 | loss   0.5466 | correct  50 | time  0.5216s
Epoch 200 | loss   0.4178 | correct  50 | time  0.9194s
Epoch 210 | loss   0.0026 | correct  50 | time  0.5261s
Epoch 220 | loss   0.5469 | correct  50 | time  0.8838s
Epoch 230 | loss   0.1436 | correct  50 | time  0.4971s
Epoch 240 | loss   0.2610 | correct  50 | time  0.4919s
Epoch 250 | loss   0.1928 | correct  50 | time  0.4902s
Epoch 260 | loss   0.0003 | correct  50 | time  0.4908s
Epoch 270 | loss   0.0028 | correct  50 | time  0.4910s
Epoch 280 | loss   0.0578 | correct  50 | time  0.4984s
Epoch 290 | loss   0.0644 | correct  50 | time  0.4955s
Epoch 300 | loss   0.0784 | correct  50 | time  0.4918s
Epoch 310 | loss   0.0620 | correct  50 | time  0.7317s
Epoch 320 | loss   0.8042 | correct  50 | time  0.4959s
Epoch 330 | loss   0.4551 | correct  50 | time  0.9169s
Epoch 340 | loss   0.0043 | correct  50 | time  0.4921s
Epoch 350 | loss   0.3771 | correct  50 | time  0.4931s
Epoch 360 | loss   0.0645 | correct  50 | time  0.4938s
Epoch 370 | loss   0.0098 | correct  50 | time  0.4930s
Epoch 380 | loss   0.3878 | correct  50 | time  0.4905s
Epoch 390 | loss   0.0050 | correct  50 | time  0.4927s
Epoch 400 | loss   0.4370 | correct  50 | time  0.4887s
Epoch 410 | loss   0.0494 | correct  50 | time  0.5029s
Epoch 420 | loss   0.0936 | correct  50 | time  0.4967s
Epoch 430 | loss   0.0003 | correct  50 | time  0.4960s
Epoch 440 | loss   0.0942 | correct  50 | time  0.8925s
Epoch 450 | loss   0.4659 | correct  50 | time  0.4873s
Epoch 460 | loss   0.0008 | correct  50 | time  0.5817s
Epoch 470 | loss   0.1648 | correct  50 | time  0.4937s
Epoch 480 | loss   0.0749 | correct  50 | time  0.4905s
Epoch 490 | loss   0.0253 | correct  50 | time  0.4940s
Epoch 500 | loss   0.0491 | correct  50 | time  0.5847s

Average epoch time: 0.5847s
```
</details>

### Big Model (200 hidden layer size)

<details>
<summary>GPU on Simple Data Training Logs Dropdown</summary>

```
Epoch   0 | loss   6.9112 | correct  45 | time  4.0605s
Epoch  10 | loss   1.9984 | correct  48 | time  1.9339s
Epoch  20 | loss   1.2149 | correct  49 | time  2.2495s
Epoch  30 | loss   1.0125 | correct  47 | time  1.9061s
Epoch  40 | loss   1.8280 | correct  46 | time  1.9199s
Epoch  50 | loss   0.2450 | correct  50 | time  2.6746s
Epoch  60 | loss   1.0769 | correct  50 | time  1.9174s
Epoch  70 | loss   0.9866 | correct  49 | time  1.9170s
Epoch  80 | loss   0.9234 | correct  49 | time  2.1881s
Epoch  90 | loss   0.0479 | correct  47 | time  1.9110s
Epoch 100 | loss   0.0045 | correct  49 | time  1.9858s
Epoch 110 | loss   0.2139 | correct  50 | time  2.0066s
Epoch 120 | loss   0.6052 | correct  49 | time  1.9998s
Epoch 130 | loss   0.4634 | correct  49 | time  1.9976s
Epoch 140 | loss   0.9338 | correct  49 | time  1.9194s
Epoch 150 | loss   2.5701 | correct  49 | time  2.2055s
Epoch 160 | loss   0.1670 | correct  49 | time  1.9208s
Epoch 170 | loss   1.4899 | correct  49 | time  1.9167s
Epoch 180 | loss   0.1878 | correct  50 | time  2.6387s
Epoch 190 | loss   1.1364 | correct  49 | time  1.9322s
Epoch 200 | loss   0.0536 | correct  50 | time  1.9119s
Epoch 210 | loss   1.0939 | correct  49 | time  2.3249s
Epoch 220 | loss   0.3344 | correct  50 | time  1.9323s
Epoch 230 | loss   1.3118 | correct  49 | time  1.9307s
Epoch 240 | loss   0.0003 | correct  50 | time  2.0205s
Epoch 250 | loss   0.0848 | correct  49 | time  1.9183s
Epoch 260 | loss   0.0096 | correct  49 | time  1.9632s
Epoch 270 | loss   1.5927 | correct  49 | time  2.6544s
Epoch 280 | loss   0.3091 | correct  49 | time  2.4850s
Epoch 290 | loss   0.0058 | correct  49 | time  1.8912s
Epoch 300 | loss   1.1198 | correct  49 | time  1.8995s
Epoch 310 | loss   0.0148 | correct  49 | time  2.7353s
Epoch 320 | loss   1.4204 | correct  49 | time  1.9101s
Epoch 330 | loss   0.1418 | correct  49 | time  1.9234s
Epoch 340 | loss   0.0968 | correct  49 | time  2.6751s
Epoch 350 | loss   1.4045 | correct  49 | time  1.9103s
Epoch 360 | loss   0.2198 | correct  49 | time  1.9064s
Epoch 370 | loss   0.4876 | correct  49 | time  2.4644s
Epoch 380 | loss   1.2692 | correct  49 | time  1.9094s
Epoch 390 | loss   0.0441 | correct  49 | time  1.9127s
Epoch 400 | loss   0.0220 | correct  49 | time  2.1484s
Epoch 410 | loss   1.3763 | correct  49 | time  1.9640s
Epoch 420 | loss   0.0104 | correct  50 | time  1.9631s
Epoch 430 | loss   0.1029 | correct  50 | time  2.0079s
Epoch 440 | loss   0.4727 | correct  49 | time  1.9252s
Epoch 450 | loss   1.0824 | correct  49 | time  1.9218s
Epoch 460 | loss   0.0005 | correct  50 | time  1.8918s
Epoch 470 | loss   0.3888 | correct  49 | time  2.0757s
Epoch 480 | loss   0.3617 | correct  49 | time  2.0990s
Epoch 490 | loss   0.0034 | correct  49 | time  1.9098s
Epoch 500 | loss   0.0235 | correct  49 | time  2.0774s

Average epoch time: 2.0774s
```
</details>

<details>
<summary>CPU on Simple Data Training Logs Dropdown</summary>

```
Epoch   0 | loss   4.5880 | correct  45 | time 15.9540s
Epoch  10 | loss   0.9960 | correct  47 | time  1.8326s
Epoch  20 | loss   2.9035 | correct  44 | time  1.8153s
Epoch  30 | loss   0.5674 | correct  50 | time  2.4777s
Epoch  40 | loss   0.5271 | correct  50 | time  1.8160s
Epoch  50 | loss   0.8149 | correct  47 | time  1.8203s
Epoch  60 | loss   0.1483 | correct  50 | time  1.8784s
Epoch  70 | loss   0.5274 | correct  50 | time  1.8556s
Epoch  80 | loss   1.0523 | correct  50 | time  2.4872s
Epoch  90 | loss   0.1777 | correct  50 | time  1.8101s
Epoch 100 | loss   0.3004 | correct  50 | time  1.8060s
Epoch 110 | loss   0.0365 | correct  50 | time  2.9012s
Epoch 120 | loss   0.0038 | correct  50 | time  1.8236s
Epoch 130 | loss   0.8593 | correct  50 | time  1.8293s
Epoch 140 | loss   0.0311 | correct  50 | time  1.9620s
Epoch 150 | loss   0.1554 | correct  50 | time  1.8191s
Epoch 160 | loss   0.2032 | correct  50 | time  1.8710s
Epoch 170 | loss   0.2541 | correct  50 | time  1.8166s
Epoch 180 | loss   0.4123 | correct  50 | time  1.8180s
Epoch 190 | loss   0.1417 | correct  50 | time  2.0601s
Epoch 200 | loss   0.1261 | correct  50 | time  1.8239s
Epoch 210 | loss   0.2982 | correct  50 | time  1.8294s
Epoch 220 | loss   0.4495 | correct  50 | time  2.9129s
Epoch 230 | loss   0.0062 | correct  50 | time  1.8375s
Epoch 240 | loss   0.1837 | correct  50 | time  1.8470s
Epoch 250 | loss   0.3230 | correct  50 | time  2.4935s
Epoch 260 | loss   0.0100 | correct  50 | time  1.8514s
Epoch 270 | loss   0.0761 | correct  50 | time  1.8455s
Epoch 280 | loss   0.0022 | correct  50 | time  1.8669s
Epoch 290 | loss   0.0065 | correct  50 | time  1.9763s
Epoch 300 | loss   0.0948 | correct  50 | time  1.8112s
Epoch 310 | loss   0.2823 | correct  50 | time  1.8076s
Epoch 320 | loss   0.0215 | correct  50 | time  1.8248s
Epoch 330 | loss   0.0194 | correct  50 | time  2.3365s
Epoch 340 | loss   0.0071 | correct  50 | time  1.8324s
Epoch 350 | loss   0.0989 | correct  50 | time  1.8433s
Epoch 360 | loss   0.2909 | correct  50 | time  2.8891s
Epoch 370 | loss   0.2836 | correct  50 | time  1.8056s
Epoch 380 | loss   0.0250 | correct  50 | time  1.8194s
Epoch 390 | loss   0.0401 | correct  50 | time  1.8732s
Epoch 400 | loss   0.0001 | correct  50 | time  1.8129s
Epoch 410 | loss   0.0208 | correct  50 | time  1.9922s
Epoch 420 | loss   0.0124 | correct  50 | time  1.8213s
Epoch 430 | loss   0.1308 | correct  50 | time  1.8326s
Epoch 440 | loss   0.1476 | correct  50 | time  2.8854s
Epoch 450 | loss   0.1109 | correct  50 | time  1.8061s
Epoch 460 | loss   0.0406 | correct  50 | time  1.8024s
Epoch 470 | loss   0.1709 | correct  50 | time  2.4601s
Epoch 480 | loss   0.0323 | correct  50 | time  1.8264s
Epoch 490 | loss   0.0411 | correct  50 | time  1.8168s
Epoch 500 | loss   0.1413 | correct  50 | time  2.0388s

Average epoch time: 2.0388s
```
</details>


