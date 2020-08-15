"""
Tests for numpy implementation of pair energy.
"""
import pytest
import os 
import math
import numpy as np
import calculate_pair_energy_np

def test_calculate_pair_energy_np():
    # test NIST data
    filepath = os.path.join('..', '..','lj_sample_configurations', 'lj_sample_config_periodic1.txt')
    coordinates, box_length = calculate_pair_energy_np.read_xyz(filepath)
    expected_value = -10.877945969430789 #number retrieved from PSL implementation
    observed_value = calculate_pair_energy_np.calculate_pair_energy_np(coordinates, 0, box_length, 3)
    assert math.isclose(expected_value, observed_value, rel_tol = 0.001)

def test_total_energy():
    expected_value = -2.031005859375
    atomic_coordinates = np.array([[0,0,0],[0,0,2**(1/6)],[0,0,2*(2**(1/6))]])
    observed_value = calculate_pair_energy_np.calculate_total_energy_np(atomic_coordinates, 10, 3)
    assert expected_value == observed_value

def test_total_energy2():
    expected_value = -2
    atomic_coordinates = np.array([[0,0,0],[0,0,2**(1/6)],[0,0,2*(2**(1/6))]])
    observed_value = calculate_pair_energy_np.calculate_total_energy_np(atomic_coordinates, 10, 2)
    assert expected_value == observed_value
    