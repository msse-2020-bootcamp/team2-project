#include <array> // AtomCoord
#include <vector> // Coordinates
#include <math.h> //pow, round
#include <cstdlib> //absolute value
#include <iostream> //print
#include <random> // for random numbers
#include <chrono> // for generating the seed
#include <string> // strings
#include <fstream> // reading / writing files

typedef std::array<double,3> AtomCoord;
typedef std::vector<AtomCoord> Coordinates;
std::default_random_engine re;

double random_double(double lower_bound, double upper_bound)
{
   std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
   return dist(re);
}

double random_int(int lower_bound, int upper_bound)
{           
   //dist will return [a,b] but we want [a,b)
   std::uniform_int_distribution<int> dist(lower_bound, upper_bound-1);
   return dist(re);
}  

double calculate_LJ(double r_ij) {
    double inverse = 1.0/r_ij;
    double pairwise_energy = 4.0 * pow(inverse, 12.0) - pow(inverse, 6.0);
    return pairwise_energy;
}

double calculate_distance(AtomCoord coord1, AtomCoord coord2, double box_length = INFINITY){
    double distance = 0;
    for (int i = 0; i < sizeof(coord1)/sizeof(coord1[0]); i++) {
        double coord_dist = coord1[i] - coord2[i];
        if (box_length < INFINITY){
            if (abs(coord_dist) > box_length/2.0) {
                coord_dist = coord_dist - (box_length * round(coord_dist/box_length));
            }
        }
        distance += pow(coord_dist,2);
    }
    distance = sqrt(distance);
    return distance;
} 

double calculate_total_energy(Coordinates coords, double box_length, double cutoff){
    double total_energy = 0;

    for (int i = 0; i < coords.size(); i++) {
        for (int j = i+1; j < coords.size(); j++) {
            double distance = calculate_distance(coords.at(i), coords.at(j), box_length);
            if (distance < cutoff) {
                double energy = calculate_LJ(distance);
                total_energy += energy;
            }
        }
    }
    return total_energy;
}

double calculate_tail_correction(int num_particles, double box_length, double cutoff) {
    double const1 = (8.0 * M_PI * pow(num_particles, 2)) / (3.0 * pow(box_length,3));
    double const2 = (1.0/3.0) * pow(1.0/cutoff, 9) - pow(1.0/cutoff,3);
    return const1 * const2;
}

std::pair<Coordinates, double> read_xyz(std::string file_path)
{
    // Opens up a file stream for input
    std::ifstream infile(file_path);

    // Check that it was successfully opened
    if(!infile.is_open())
    {   
        throw std::runtime_error("File path in read_xyz does not exist!");
    }
    
    double dummy; // Data that is thrown away (box length, atom indices)
    double box_length;
    int num_atoms;
    
    // Grab box_length from first number, throw the rest away
    infile >> box_length >> dummy >> dummy;
    
    // now the number of atoms
    infile >> num_atoms;
    
    // Uncomment to help troubleshoot
    //std::cout << "Box length: " << box_length << " natoms: " << num_atoms << std::endl;
    
    // Stores the atomic coordinates
    // Remember, this is a vector of arrays
    Coordinates coords;
    
    for(int i = 0; i < num_atoms; i++)
    {   
        AtomCoord coord;
        
        // Throws away the atom index
        infile >> dummy >> coord[0] >> coord[1] >> coord[2];
        
        // Add to the vector
        coords.push_back(coord);
    }

    // Makes an appropriate pair object
    return std::make_pair(coords, box_length);
}

std::pair<Coordinates, double> init_config(int num_particles, double box_length) {
    Coordinates coords;
    coords.resize(num_particles);

    for (int i = 0; i < num_particles; i++) {
        AtomCoord a = {random_double(0.0, box_length), random_double(0.0, box_length), random_double(0.0, box_length)};
        coords.at(i) = a;
    }

    return std::make_pair(coords, box_length);
}

bool accept_or_reject(double delta_e, double beta) {
    bool accept;
    if (delta_e == 0) {
        accept = true;
    } else {
        double random_number = random_double(0,1);
        double p_acc = exp(-beta * delta_e);
        if (random_number < p_acc) {
            accept = true;
        } else {
            accept = false;
        }
    }
    return accept;
}

double calculate_pair_energy(Coordinates coords, int i_particle, double box_length, double cutoff) {
    double e_total = 0;
    for (int i =0; i < coords.size(); i++) {
        if (i != i_particle) {
            double dist = calculate_distance(coords[i], coords[i_particle], box_length);
            if (dist < cutoff) {
                e_total += calculate_LJ(dist);
            }
        }
    }
    return e_total;
}

Coordinates run_simulation(Coordinates coords, double box_length, double cutoff, 
                           double reduced_temperature, int num_steps, double max_displacement =0.1, int freq = 1000) {
    int steps[num_steps / freq];
    double energies[num_steps / freq];

    double beta = 1.0/reduced_temperature;
    int num_particles = coords.size();

    double total_energy = calculate_total_energy(coords, box_length, cutoff);
    total_energy += calculate_tail_correction(num_particles, box_length, cutoff);

    for (int step = 0; step < num_steps; step++) {
        int random_particle = random_int(0, num_particles);
        double current_energy = calculate_pair_energy(coords, random_particle, box_length, cutoff);

        double x_rand = random_double(-max_displacement, max_displacement);
        double y_rand = random_double(-max_displacement, max_displacement);
        double z_rand = random_double(-max_displacement, max_displacement);
        
        coords[random_particle][0] += x_rand;
        coords[random_particle][1] += y_rand;
        coords[random_particle][2] += z_rand;
        
        double proposed_energy = calculate_pair_energy(coords, random_particle, box_length, cutoff);

        double delta_energy = proposed_energy - current_energy;
        bool accept = accept_or_reject(delta_energy, beta);

        if (accept) {
            total_energy += delta_energy;
        } else {
            coords[random_particle][0] -= x_rand;
            coords[random_particle][1] -= y_rand;
            coords[random_particle][2] -= z_rand;
        }

        if (step % freq == 0) {
            std::cout << step << " " << total_energy/num_particles << std::endl;
            steps[static_cast<int> (floor(step/freq))] = step;
            energies[(int) floor(step/freq)] = total_energy/num_particles;
        }
    }
    return coords;
}

int main(void) {
    // Initialize random number generation based on time
    re.seed(std::chrono::system_clock::now().time_since_epoch().count());

    std::pair<Coordinates, double> xyz_info = read_xyz("../../lj_sample_configurations/lj_sample_config_periodic1.txt");
    Coordinates coords = xyz_info.first;
    double box_length = xyz_info.second;
    Coordinates new_coords = run_simulation(coords, box_length, 3.0, 0.9, 5000);
    return 0;
}