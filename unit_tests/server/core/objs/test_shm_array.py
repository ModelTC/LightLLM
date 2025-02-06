# test_shm_array.py

import pytest
import numpy as np
from multiprocessing import shared_memory
from lightllm.server.core.objs.shm_array import ShmArray  # Replace 'your_module' with the actual module name


@pytest.fixture(scope="module")
def shm_array():
    name = "test_shm_array"
    shape = (10, 10)
    dtype = np.float32
    shm = ShmArray(name, shape, dtype)
    shm.create_shm()
    yield shm
    shm.close_shm()


def test_create_shm(shm_array):
    assert shm_array.shm is not None
    assert shm_array.arr.shape == shm_array.shape
    assert shm_array.arr.dtype == shm_array.dtype


def test_link_shm(shm_array):
    # Create a new instance to link the existing shm
    shm_linked = ShmArray(shm_array.name, shm_array.shape, shm_array.dtype)
    shm_linked.link_shm()

    assert shm_linked.shm is not None
    assert shm_linked.arr.shape == shm_array.shape
    assert shm_linked.arr.dtype == shm_array.dtype


def test_shm_size_mismatch(shm_array):
    # Create a new shm with a different size
    shape_mismatch = (5, 5)
    shm_mismatch = ShmArray(shm_array.name, shape_mismatch, shm_array.dtype)

    with pytest.raises(Exception):
        shm_mismatch.link_shm()


def test_close_shm(shm_array):
    shm_array.close_shm()
    assert shm_array.shm is None


def test_recreate_shm_after_close(shm_array):
    shm_array.close_shm()

    # Attempting to create a new shm after close
    shm_new = ShmArray(shm_array.name, shm_array.shape, shm_array.dtype)
    shm_new.create_shm()

    assert shm_new.shm is not None
    assert shm_new.arr.shape == shm_array.shape
    assert shm_new.arr.dtype == shm_array.dtype

    shm_new.close_shm()


if __name__ == "__main__":
    pytest.main()
