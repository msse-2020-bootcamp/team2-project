"""
Tests for Python Standard Library implementation of MC code.
"""
import math
import pytest
from monte_carlo import calculate_distance
from monte_carlo import calculate_LJ

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
    observed_distance = calculate_distance(point1, point2, box_length)
    assert expected_distance == observed_distance

# @pytest.mark.skip
def test_calculate_LJ():
    expected_value = 0
    observed_value = calculate_LJ(1)
    assert expected_value == observed_value

    # expected_value = -1
    # observed_value = calculate_LJ(math.pow(2, 1/6))
    # assert expected_value == observed_value

def test_calculate_distLJ_error():
    with pytest.raises(ZeroDivisionError):
        calculate_LJ(0)