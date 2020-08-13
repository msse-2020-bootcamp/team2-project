"""
Tests for Python Standard Library implementation of MC code.
"""
import math
import pytest
import random
import os
import monte_carlo as mc

"""
@pytest.mark.parametrize('var_name1, var_name2, ..., var_nameN', [
    (variable_val1, variable_val2, ..., variable_valN),
    (variable_val1, variable_val2, ..., variable_valN),
]
)
def test_function(var_name1, var_name2, ..., var_nameN):
    ***TEST CODE HERE***
"""

@pytest.mark.parametrize("point1, point2, expected_distance, box_length", [
    ([0,0,0], [0,0,8], 2, 10),
    ([0,0,0], [0,1,1], math.sqrt(2), None),
    ([0,0,0], [0,0,8], 8, None)
]
)
def test_calculate_distance(point1, point2, expected_distance, box_length):
    observed_distance = mc.calculate_distance(point1, point2, box_length)
    assert expected_distance == observed_distance

@pytest.mark.parametrize("r_ij, exp_LJ", [
    (1, 0),
    (math.pow(2, 1/6), -1),
]
)
def test_calculate_LJ(r_ij, exp_LJ):
    observed_value = mc.calculate_LJ(r_ij)
    assert exp_LJ == observed_value

@pytest.mark.parametrize("seed, del_e, beta, exp_bool", [
    (0, 1, 1, False),
    (1, 1, 1, True),
]
)
def test_accept_reject(seed, del_e, beta, exp_bool):
    random.seed(seed)
    observed_bool = mc.accept_or_reject(del_e, beta)
    random.seed()
    assert exp_bool == observed_bool

def test_pair_energy():
    coordinates = [[0,0,0], [0,0,2**(1/6)], [0,0,2*(2**(1/6))]]
    assert mc.calculate_pair_energy(coordinates, 1, 10, 3) == -2

def test_total_energy_nist():
    """
    Integration test that calculated correct value for a file in NIST
    """
    exp_val = -4.3515E+03

    filepath = os.path.join('..', '..', 'lj_sample_configurations', 'lj_sample_config_periodic1.txt')
    coords, box_length = mc.read_xyz(filepath)
    obs_val = mc.calculate_total_energy(coords, box_length, 3)
    assert math.isclose(exp_val, obs_val, rel_tol = 0.01)