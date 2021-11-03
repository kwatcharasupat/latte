from latte.functional.disentanglement import mutual_info as mi
import numpy as np
import pytest

def test_validate_za_shape_bad_samples():
    
    with pytest.raises(AssertionError):
        mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(15, 2))

def test_validate_za_shape_bad_attributes():
    
    with pytest.raises(AssertionError):
        mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(16, 33))

def test_validate_za_shape_bad_zdim():
    
    with pytest.raises(AssertionError):
        mi._validate_za_shape(np.random.randn(16, 32, 3), np.random.randn(16, 32))

def test_validate_za_shape_bad_adim():
    
    with pytest.raises(AssertionError):
        mi._validate_za_shape(np.random.randn(16, 32), np.random.randn(16, 32, 3))