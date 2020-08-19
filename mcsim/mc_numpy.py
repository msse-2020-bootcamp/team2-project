import numpy as np
import math
import random

def init_config(num_particles, density):
    """
    Generates initial system configuration from a number of particles and box length.
    
    Parameters:
    ```````````
    num_particles : int
        The number of particles desired in configuration.
        
    density : float
        The reduced density of the system.
    
    Returns:
    ````````
    coordinates : np.ndarray
        Numpy array of coordinates [x,y,z] in configuration.
        
    box_length : float
        The length of the simulation box.
    
    """
    coordinates = np.zeros((1, 3))
    volume = num_particles / density
    box_length = volume ** (1/3)
    for i in range(num_particles):
        # generate random coordinates
        x = random.uniform(0, box_length)
        y = random.uniform(0, box_length)
        z = random.uniform(0, box_length)
        
        # add (x, y, z) to the list
        coordinates = np.vstack((coordinates, np.array([[x, y, z]])))
    
    return coordinates, box_length

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
    if delta_e <= 0:
        accept = True
    else:
        random_number = random.random()
        p_acc = math.exp(-beta * delta_e)
        
        if random_number < p_acc:
            accept = True
        else:
            accept = False
    
    return accept

def calculate_tail_correction(num_particles, box_length, cutoff):
    """
    Calculate the long range tail correction
    """
    
    const1 = (8 * math.pi * num_particles ** 2) / (3 * box_length ** 3)
    const2 = (1/3) * (1 / cutoff)**9 - (1 / cutoff) **3
    
    return const1 * const2

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

def rdf(values, n_bins, max_value, num_particles, box_length):
    """
    Compute the RDF for a set of values
    
    Parameters
    -----------
    values : np.ndarray
        The distance values to compute the RDF for.
        
    n_bins : int
        the number of bins to use for the histogram
    
    max_value : float
        The maximum value for which to compute the RDF.
    
    num_particles : int
        The number of particles in the system.
    
    box_length : float
        The box length
    
    Returns
    -------
    bin_centers : np.ndarray
        An array of the bin centers
        
    rdf : np.ndarray
        An array of the rdf values.
    
    """
    
    histogram, bins = np.histogram(values, bins=n_bins, range=(0, max_value))
    
    bin_size = bins[1] - bins[0]
    
    bin_centers = bins + bin_size/2
    bin_centers = bin_centers[:-1]
    
    rdf = []
    
    rdf = histogram / (4 * math. pi * bin_centers**2 * bin_size * num_particles ** 2 / box_length ** 3)
    
    return bin_centers, rdf

def run_sim(reduced_temperature, reduced_density, num_steps=1000000,
            max_displacement=0.1, cutoff=3, num_particles = 500, freq=100):
    # set simulation parameters
    rdf_lst = []

    # reporting information
    steps = np.zeros((math.floor(num_steps/freq), 1))
    energies = np.zeros((math.floor(num_steps/freq), 1))

    # calculated quantities
    beta = 1 / reduced_temperature

    coordinates, box_length = init_config(num_particles, reduced_density)

    total_energy = calculate_total_energy_np(coordinates, box_length, cutoff)
    total_energy += calculate_tail_correction(num_particles, box_length, cutoff)

    for step in range(num_steps):
        # 1. randomly pick one of num_particles with uniform distribution
        random_particle = random.randrange(0, num_particles)
        
        # 2. calculate the interaction energy of selected particle with system.
        current_energy = calculate_pair_energy_np(coordinates, random_particle, box_length, cutoff)
        
        # 3. generate random x, y, z displacement range (-max_displacement, max_displacement)
        x_rand = random.uniform(-max_displacement, max_displacement)
        y_rand = random.uniform(-max_displacement, max_displacement)
        z_rand = random.uniform(-max_displacement, max_displacement)
        
        # 4. modify coordinate of selected particle by generated displacements
        coordinates[random_particle][0] += x_rand
        coordinates[random_particle][1] += y_rand
        coordinates[random_particle][2] += z_rand
        
        # 5. calculate new interaction energy of moved particle.
        proposed_energy = calculate_pair_energy_np(coordinates, random_particle, box_length, cutoff)
        
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
            steps[math.floor(step/freq)][0] = step
            energies[math.floor(step/freq)][0] = total_energy/num_particles

        if num_steps - step <= 1000 and (num_steps - step) % 100 == 0:
            values = []
            for i in range(num_particles):
                for j in range(num_particles):
                    if i != j:
                        values.append(calculate_distance_np(coordinates[i], coordinates[j], box_length))
            bins, rdfs = rdf(np.array(values), 100, box_length/2, num_particles, box_length)
            rdf_lst.append(rdfs)

    rdf_lst = np.array(rdf_lst)
    averaged_rdfs = np.mean(rdf_lst, axis=0)
    
    return steps, energies, bins, averaged_rdfs