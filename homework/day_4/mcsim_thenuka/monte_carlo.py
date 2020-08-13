"""
Functions for running Monte Carlo simulation.
"""

import math
import random
import os
import matplotlib.pyplot as plt

def calculate_LJ(r_ij):
    """
    The LJ interaction energy between two particles.
    
    Computes the pairwise Lennard-Jones interaction energy based on the separation distance in reduced units.
    
    Parameters:
    ```````````
    r_ij : float
        The separation distance in reduced units.
    
    Returns:
    ````````
    pairwise energy : float
        The pairwise Lennard-Jones interaction energy in reduced units.
    """
    
    try:
        r6_term = math.pow(1/r_ij, 6)
    except ZeroDivisionError:
        raise ZeroDivisionError("Infinite energy calculated - particles overlapping!")


    r12_term = math.pow(r6_term, 2)
    
    pairwise_energy = 4 * (r12_term - r6_term)

    return pairwise_energy


def calculate_distance(coord1, coord2, box_length = None):
    """
    Calculate the distance between two points. When box_length is set, we use the the minimum image convention to calculate.
    
    Parameters:
    ```````````
    coord1, coord2 : list
        The atomic coordinates [x, y, z]
        
    box_length : float, optional
        The box length. The function assumes the box is a cube.
    
    Returns:
    ````````
    distance : float
        The distance between the two atoms.
    """
    distance = 0
    for i in range(len(coord1)):
        coord_dist = coord1[i] - coord2[i]
        if box_length:
            if abs(coord_dist) > box_length / 2:
                coord_dist = coord_dist - (box_length * round(coord_dist / box_length))
        distance += coord_dist**2
        
    distance = math.sqrt(distance)
    
    return distance


def calculate_total_energy(coords, box_length, cutoff):
    """
    Calculates the total interaction energy existing among a set of coordinates.
    
    Parameters:
    ```````````
    coords : list
        Nested list of coordinates [x,y,z]
    cutoff : float
        The cutoff distance for the system
        
    Returns:
    ````````
    total_energy : float
        The total interaction energy calculated from LJ potential.
    """
    
    total_energy = 0
    
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = calculate_distance(coords[i], coords[j], box_length)
            if dist < cutoff:
                energy = calculate_LJ(dist)
                total_energy += energy
    
    return total_energy

def calculate_tail_correction(num_particles, box_length, cutoff):
    """
    Calculate the long range tail correction
    """
    
    const1 = (8 * math.pi * num_particles ** 2) / (3 * box_length ** 3)
    const2 = (1/3) * (1 / cutoff)**9 - (1 / cutoff) **3
    
    return const1 * const2


def read_xyz(filepath):
    """
    Reads coordinates from an xyz file.
    
    Parameters
    ----------
    filepath : str
       The path to the xyz file to be processed.
       
    Returns
    -------
    atomic_coordinates : list
        A two dimensional list containing atomic coordinates
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
        
    
    return atomic_coordinates, box_length


def init_config(num_particles, box_length):
    """
    Generates initial system configuration from a number of particles and box length.
    
    Parameters:
    ```````````
    num_particles : int
        The number of particles desired in configuration.
        
    box_length : float
        The length of the simulation box.
    
    Returns:
    ````````
    coordinates : list
        Nested list of coordinates [x,y,z] in configuration.
        
    box_length : float
        The length of the simulation box.
    
    """
    coordinates = []
    
    for i in range(num_particles):
        # generate random coordinates
        x = random.uniform(0, box_length)
        y = random.uniform(0, box_length)
        z = random.uniform(0, box_length)
        
        # add (x, y, z) to the list
        coordinates.append([x, y, z])
    
    return coordinates, box_length


def accept_or_reject(delta_e, beta):
    """
    Accept or reject a new state based on change in energy and temperature.
    
    Parameters:
    ```````````
    delta_e : float
        change in energy
    
    beta : float
        inverse of temperature
        
    Returns:
    ````````
    accept : bool
        T/F value of whether to accept change
    """
    if delta_e == 0:
        accept = True
    else:
        random_number = random.random()
        p_acc = math.exp(-beta * delta_e)
        
        if random_number < p_acc:
            accept = True
        else:
            accept = False
    
    return accept


def calculate_pair_energy(coordinates, i_particle, box_length, cutoff):
    """
    Calculates the interaction energy of one particle with all others in system.
    
    Parameters:
    ```````````
    coordinates : list
       nested list of [x,y,z] coordinates for all particles in the system
        
    i_particle : int
        the particle index for which to calculate energy
        
    box_length : float
        the length of the simulation box
        
    cutoff : float
        the cutoff interaction length
        
    Returns:
    ````````
    e_total : float
        the pairwise energy between the i-th particle and other particles in system
    """
    e_total = 0
    for i in range(len(coordinates)):
        if i == i_particle:
            continue
        dist = calculate_distance(coordinates[i], coordinates[i_particle], box_length)
        if dist < cutoff:
            energy = calculate_LJ(dist)
            e_total += energy
    
    return e_total


def run_simulation(coordinates, box_length, cutoff, reduced_temperature, num_steps, max_displacement=0.1, freq=1000):
    """
    Runs a Monte Carlo simulation with specified parameters.
    """

    steps = []
    energies = []

    # calculated quantities
    beta = 1 / reduced_temperature
    num_particles = len(coordinates)

    total_energy = calculate_total_energy(coordinates, box_length, cutoff)
    total_energy += calculate_tail_correction(num_particles, box_length, cutoff)

    for step in range(num_steps):
        # 1. randomly pick one of num_particles with uniform distribution
        random_particle = random.randrange(0, num_particles)
        
        # 2. calculate the interaction energy of selected particle with system.
        current_energy = calculate_pair_energy(coordinates, random_particle, box_length, cutoff)
        
        # 3. generate random x, y, z displacement range (-max_displacement, max_displacement)
        x_rand = random.uniform(-max_displacement, max_displacement)
        y_rand = random.uniform(-max_displacement, max_displacement)
        z_rand = random.uniform(-max_displacement, max_displacement)
        
        # 4. modify coordinate of selected particle by generated displacements
        coordinates[random_particle][0] += x_rand
        coordinates[random_particle][1] += y_rand
        coordinates[random_particle][2] += z_rand
        
        # 5. calculate new interaction energy of moved particle.
        try:
            proposed_energy = calculate_pair_energy(coordinates, random_particle, box_length, cutoff)
        except ZeroDivisionError:
            raise ZeroDivisionError("Particle overlap has occured! Halting simulation due infinite energy!")

        
        
        # 6. calculate energy change and decide to accept / reject.
        delta_energy = proposed_energy - current_energy
        accept = accept_or_reject(delta_energy, beta)
        
        # 7. if accept, keep movement. if not revert to old position
        if accept:
            total_energy += delta_energy
        else:
            coordinates[random_particle][0] -= x_rand
            coordinates[random_particle][1] -= y_rand
            coordinates[random_particle][2] -= z_rand
        
        # 8. print energy at certain intervals
        if step % freq == 0:
            print(step, total_energy/num_particles)
            steps.append(step)
            energies.append(total_energy/num_particles)

    return coordinates