import numpy as np
import pytest

from puzzles import (
    rotate_90,
    transpose_no_T,
    is_magic_square,
    block_trace,
    top_k_indices,
)


def test_rotate_90_basic():
    m = np.array([[1, 2], [3, 4]])
    expected = np.array([[3, 1], [4, 2]])
    assert np.array_equal(rotate_90(m), expected)


def test_rotate_90_single_element():
    m = np.array([[5]])
    assert np.array_equal(rotate_90(m), m)


def test_rotate_90_non_square():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.array([[4, 1], [5, 2], [6, 3]])
    assert np.array_equal(rotate_90(m), expected)


def test_transpose_no_T_basic():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    assert np.array_equal(transpose_no_T(m), m.T)


def test_transpose_no_T_single_element():
    m = np.array([[7]])
    assert np.array_equal(transpose_no_T(m), m)


def test_transpose_no_T_square():
    m = np.arange(9).reshape(3, 3)
    assert np.array_equal(transpose_no_T(m), m.T)



def test_is_magic_square_true():
    m = np.array([[2, 7, 6], [9, 5, 1], [4, 3, 8]])
    assert is_magic_square(m) is True


def test_is_magic_square_false():
    m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert is_magic_square(m) is False


def test_is_magic_square_non_square_input():
    m = np.array([[1, 2, 3], [4, 5, 6]])
    assert is_magic_square(m) is False


def test_is_magic_square_non_2d_input():
    v = np.array([1, 2, 3])
    assert is_magic_square(v) is False


def test_is_magic_square_single_element():
    m = np.array([[5]])
    assert is_magic_square(m) is True


def test_block_trace_example():
    m = np.arange(16).reshape(4, 4)
    expected = np.array([[5, 9], [21, 25]])
    assert np.array_equal(block_trace(m, 2), expected)


def test_block_trace_full_matrix_as_one_block():
    m = np.arange(9).reshape(3, 3)
    expected = np.array([[np.trace(m)]])
    assert np.array_equal(block_trace(m, 3), expected)


def test_block_trace_unit_blocks():
    m = np.arange(4).reshape(2, 2)
    expected = m.copy()
    assert np.array_equal(block_trace(m, 1), expected)



def test_top_k_indices_basic():
    v = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    result = top_k_indices(v, 3)
    assert list(result) == [5, 7, 4]


def test_top_k_indices_k_equals_length():
    v = np.array([10, 30, 20])
    result = top_k_indices(v, 3)
    assert list(result) == [1, 2, 0]


def test_top_k_indices_single_element():
    v = np.array([42])
    result = top_k_indices(v, 1)
    assert list(result) == [0]
