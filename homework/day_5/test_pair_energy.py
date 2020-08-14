"""
Tests for numpy implementation of pair energy.
"""
import pytest
import os 
import math
from calculate_pair_energy_np import calculate_pair_energy_np
from calculate_pair_energy_np import read_xyz

def test_calculate_pair_energy_np():
    # test NIST data
    filepath = os.path.join('..', '..','lj_sample_configurations', 'lj_sample_config_periodic1.txt')
    coordinates, box_length = read_xyz(filepath)
    expected_value = -10.877945969430789 #number retrieved from PSL implementation
    observed_value = calculate_pair_energy_np(coordinates, 0, box_length, 3)
    assert math.isclose(expected_value, observed_value, rel_tol = 0.001)
