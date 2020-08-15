import math
import numpy as np
import total_energy

def test_total_energy():
    expected_value = -2.031005859375
    atomic_coordinates = np.array([[0,0,0],[0,0,2**(1/6)],[0,0,2*(2**(1/6))]])
    observed_value = total_energy.calculate_total_energy_np(atomic_coordinates, 10, 3)
    assert expected_value == observed_value

def test_total_energy2():
    expected_value = -2
    atomic_coordinates = np.array([[0,0,0],[0,0,2**(1/6)],[0,0,2*(2**(1/6))]])
    observed_value = total_energy.calculate_total_energy_np(atomic_coordinates, 10, 2)
    assert expected_value == observed_value
    