import numpy as np

def rotate_90(m):
    m = np.asarray(m)
    return m[::-1].T.copy()

def transpose_no_T(m):
    m = np.asarray(m)
    return np.einsum("ij->ji", m)

def is_magic_square(m):
    m = np.asarray(m)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    n = m.shape[0]
    if n == 0:
        return True
    row_sums = m.sum(axis=1)
    col_sums = m.sum(axis=0)
    diag_sum = np.trace(m)
    anti_diag_sum = np.trace(np.fliplr(m))
    target = row_sums[0]
    return bool(
        np.all(row_sums == target)
        and np.all(col_sums == target)
        and diag_sum == target
        and anti_diag_sum == target
    )

def block_trace(m, k):
    m = np.asarray(m)
    n = m.shape[0]
    nb = n // k
    blocks = m.reshape(nb, k, nb, k).transpose(0, 2, 1, 3)
    return np.trace(blocks, axis1=2, axis2=3)

def top_k_indices(v, k):
    v = np.asarray(v)
    k = max(0, min(k, v.size))
    if k == 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-v, k - 1)[:k]
    return idx[np.argsort(-v[idx])]