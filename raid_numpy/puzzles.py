"""Vectorized matrix puzzles. No Python for-loops over array elements."""

import numpy as np


def rotate_90(m: np.ndarray) -> np.ndarray:
    """Return a copy of the 2D array m rotated 90 degrees clockwise."""
    return np.rot90(m, k=-1).copy()


def transpose_no_T(m: np.ndarray) -> np.ndarray:
    """Return the transpose of m without using .T, np.transpose, or np.swapaxes."""
    rows, cols = m.shape
    row_idx = np.arange(rows).reshape(1, rows)   # shape (1, rows)
    col_idx = np.arange(cols).reshape(cols, 1)   # shape (cols, 1)
    return m[row_idx, col_idx]


def is_magic_square(m: np.ndarray) -> bool:
    """Return True if m is a square 2D array whose row sums, column sums,
    and both diagonals are all equal. False for non-square/non-2D input
    (must not raise)."""
    if m.ndim != 2:
        return False
    n, k = m.shape
    if n != k or n == 0:
        return False

    row_sums = m.sum(axis=1)
    col_sums = m.sum(axis=0)
    main_diag = np.trace(m)
    anti_diag = np.trace(np.fliplr(m))

    target = row_sums[0]
    return bool(
        np.all(row_sums == target)
        and np.all(col_sums == target)
        and main_diag == target
        and anti_diag == target
    )


def block_trace(m: np.ndarray, k: int) -> np.ndarray:
    """Split an n x n array into k x k blocks and return a 2D array of
    block traces."""
    n = m.shape[0]
    num_blocks = n // k
    # (n, n) -> (num_blocks, k, num_blocks, k)
    reshaped = m.reshape(num_blocks, k, num_blocks, k)
    # trace of each block = sum over the diagonal within each k x k block
    # move to (num_blocks, num_blocks, k, k) then take diagonal + sum
    blocks = reshaped.transpose(0, 2, 1, 3)  # (num_blocks, num_blocks, k, k)
    return np.trace(blocks, axis1=2, axis2=3)


def top_k_indices(v: np.ndarray, k: int) -> np.ndarray:
    """Return the indices of the k largest elements of the 1D array v,
    sorted by value descending."""
    if k <= 0:
        return np.array([], dtype=int)
    # argpartition gives the k largest in the last k positions (unordered)
    partitioned = np.argpartition(v, -k)[-k:]
    # sort just those k indices by value, descending
    order = np.argsort(v[partitioned])[::-1]
    return partitioned[order]