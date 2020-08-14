import numpy as np
import math
import os

def calculate_distance_np(point1, point2, box_length = None):
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
    dim_dist = point1 - point2
    if box_length:
        dim_dist = dim_dist - box_length*np.round(dim_dist/box_length)
    
    dim_dist = dim_dist ** 2
    if dim_dist.ndim < 2:
        dim_dist = dim_dist.reshape(1, -1)


    distance = np.sqrt(dim_dist.sum(axis=1))
    return distance

def calculate_LJ_np(r_ij):
    """
    The LJ interaction energy between two particles.
    
    Computes the pairwise Lennard-Jones interaction energy based on the separation distance in reduced units.
    
    Parameters:
    ```````````
    r_ij : np.ndarray
        The separation distance in reduced units.
    
    Returns:
    ````````
    pairwise energy : float
        The pairwise Lennard-Jones interaction energy in reduced units.
    """
    
    inverse = 1/r_ij
    pairwise_energy = 4 *(inverse**12 - inverse**6)
    return pairwise_energy


def read_xyz(filepath):
    """
    Reads coordinates from an xyz file.
    
    Parameters
    ----------
    filepath : str
       The path to the xyz file to be processed.
       
    Returns
    -------
    atomic_coordinates : np.ndarray
        A two dimensional numpy array containing atomic coordinates
    """
    
    with open(filepath) as f:
        box_length = float(f.readline().split()[0])
        num_atoms = float(f.readline())
        coordinates = f.readlines()
    
    atomic_coordinates = []
    
    for atom in coordinates:
        split_atoms = atom.split()
        
        float_coords = []
        
        # We split this way to get rid of the atom label.
        for coord in split_atoms[1:]:
            float_coords.append(float(coord))
            
        atomic_coordinates.append(float_coords)
    
    atomic_coordinates = np.array(atomic_coordinates)
    
    return atomic_coordinates, box_length

def calculate_pair_energy_np(coordinates, i_particle, box_length, cutoff):
    """
    Calculates the interaction energy of one particle with all others in system.
    
    Parameters:
    ```````````
    coordinates : np.ndarray
       2D array of [x,y,z] coordinates for all particles in the system
        
    i_particle : int
        the particle row for which to calculate energy
        
    box_length : float
        the length of the simulation box
        
    cutoff : float
        the cutoff interaction length
        
    Returns:
    ````````
    e_total : float
        the pairwise energy between the i-th particle and other particles in system
    """

    particle = coordinates[i_particle][:]
    coordinates = np.delete(coordinates, i_particle, 0)
    e_array = np.zeros(coordinates.shape)

    dist = calculate_distance_np(particle, coordinates, box_length)
    e_array = dist[dist < cutoff]
    e_array = calculate_LJ_np(e_array)
    
    e_total = e_array.sum()
    
    return e_total