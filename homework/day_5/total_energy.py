import math
import numpy as np

def calculate_LJ_np(r_ij):
    """
    The LJ interaction energy between two particles.
    
    Computes the pairwise Lennard-Jones interaction energy based on the separation distance in reduced units.
    
    Parameters
    ----------
    r_ij : float
        The separation distance in reduced units.
    
    Returns
    -------
    pairwise energy : float
        The pairwise Lennard-Jones interaction energy in reduced units.
    """
    
    try:
        r6_term = (1/r_ij)**6
    except ZeroDivisionError:
        raise ZeroDivisionError("Infinite energy calculated - particles overlapping!")


    r12_term = r6_term**2
    
    pairwise_energy = 4 * (r12_term - r6_term)

    return pairwise_energy

def calculate_distance_np(coord1, coord2, box_length=None):
    """
    Calculate the distance between two 3D coordinates.
    Parameters
    ----------
    coord1, coord2: np.ndarray
        The atomic coordinates

    Returns
    -------
    distance: float
        The distance between the two points.
    """
    coord_dist = coord1 - coord2

    if box_length:
        coord_dist = coord_dist - box_length * np.round(coord_dist / box_length)

    if coord_dist.ndim < 2:
        coord_dist = coord_dist.reshape(1, -1)

    coord_dist = coord_dist ** 2

    coord_dist_sum = coord_dist.sum(axis=1)

    distance = np.sqrt(coord_dist_sum)

    return distance

def calculate_total_energy_np(coords, box_length, cutoff):
    """
    Calculates the total interaction energy existing among a set of coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Nested array of coordinates [x,y,z]
    cutoff : float
        The cutoff distance for the system
        
    Returns
    -------
    total_energy : float
        The total interaction energy calculated from LJ potential.
    """
    
    
    total_energy = 0
    for i in range(coords.shape[0]):
        arr1 = coords[i]
        arr2 = coords[i+1:] 
        dist = calculate_distance_np(arr1, arr2, box_length)
        
    
        energy_array = dist[dist < cutoff]
        total_array = calculate_LJ_np(energy_array)
    
        total_energy += total_array.sum()
    
    return total_energy
