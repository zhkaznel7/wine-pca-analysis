import numpy as np
import pytest

from puzzles import (
    rotate_90,
    transpose_no_T,
    is_magic_square,
    block_trace,
    top_k_indices,
)


# ---------- rotate_90 ----------

def test_rotate_90_basic():
    m = np.array([[1, 2], [3, 4]])
    expected = np.array([[3, 1], [4, 2]])
    assert np.array_equal(rotate_90(m), expected)


def test_rotate_90_single_element():
    m = np.array([[5]])
    assert np.array_equal(rotate_90(m), np.array([[5]]))


def test_rotate_90_does_not_mutate_input():
    m = np.array([[1, 2], [3, 4]])
    original = m.copy()
    rotate_90(m)
    assert np.array_equal(m, original)


def test_rotate_90_returns_copy_not_view():
    m = np.array([[1, 2], [3, 4]])
    result = rotate_90(m)
    result[0, 0] = 999
    assert m[1, 0] != 999  # changing result must not affect m


# ---------- transpose_no_T ----------

def test_transpose_no_T_square():
    m = np.array([[1, 2], [3, 4]])
    expected = np.array([[1, 3], [2, 4]])
    assert np.array_equal(transpose_no_T(m), expected)


def test_transpose_no_T_rectangular():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    assert np.array_equal(transpose_no_T(m), expected)


def test_transpose_no_T_single_row():
    m = np.array([[1, 2, 3]])
    expected = np.array([[1], [2], [3]])
    assert np.array_equal(transpose_no_T(m), expected)


# ---------- is_magic_square ----------

def test_is_magic_square_true():
    m = np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]])
    assert is_magic_square(m) is True


def test_is_magic_square_false():
    m = np.array([[1, 2], [3, 4]])
    assert is_magic_square(m) is False


def test_is_magic_square_non_square_does_not_raise():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    assert is_magic_square(m) is False


def test_is_magic_square_non_2d_does_not_raise():
    v = np.array([1, 2, 3])
    assert is_magic_square(v) is False


# ---------- block_trace ----------

def test_block_trace_example():
    m = np.arange(16).reshape(4, 4)
    expected = np.array([[5, 9], [21, 25]])
    assert np.array_equal(block_trace(m, 2), expected)


def test_block_trace_k_equals_n():
    m = np.array([[1, 2], [3, 4]])
    # one block == whole matrix, trace = 1 + 4 = 5
    expected = np.array([[5]])
    assert np.array_equal(block_trace(m, 2), expected)


def test_block_trace_k_equals_1():
    m = np.array([[1, 2], [3, 4]])
    # each element is its own block, trace of a 1x1 block = the element itself
    expected = np.array([[1, 2], [3, 4]])
    assert np.array_equal(block_trace(m, 1), expected)


# ---------- top_k_indices ----------

def test_top_k_indices_basic():
    v = np.array([10, 50, 20, 40, 30])
    result = top_k_indices(v, 3)
    assert list(result) == [1, 3, 4]  # values 50, 40, 30


def test_top_k_indices_k_equals_length():
    v = np.array([3, 1, 2])
    result = top_k_indices(v, 3)
    assert list(result) == [0, 2, 1]  # values 3, 2, 1


def test_top_k_indices_k_equals_1():
    v = np.array([5, 9, 2])
    result = top_k_indices(v, 1)
    assert list(result) == [1]


def test_top_k_indices_with_duplicates():
    v = np.array([1, 5, 5, 2])
    result = top_k_indices(v, 2)
    # both 5s are the top two values, either order of the two 5-indices is fine
    assert set(result) == {1, 2}
    assert v[result[0]] == 5 and v[result[1]] == 5