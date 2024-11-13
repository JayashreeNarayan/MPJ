import csv

import numpy as np


def generate_bcc_positions(box_size, num_particles, epsilon=0.2):
    """
    Generate BCC lattice positions within a 3D box, ensuring no particles are on the boundaries.
    
    Parameters:
        box_size (float): Size of the cubic box along one dimension (assuming a cube).
        num_particles (int): Number of particles to place in the box.
        epsilon (float): Small margin to avoid placing particles on the boundaries.
    
    Returns:
        positions (np.ndarray): Array of shape (N, 3) containing the positions of the particles.
    """
    
    # Calculate the approximate number of unit cells required in each dimension
    # Each BCC unit cell contains 2 atoms
    num_cells = int(np.ceil((num_particles / 2) ** (1 / 3)))
    
    # Calculate the lattice constant (size of one unit cell)
    lattice_constant = (box_size - 2 * epsilon) / num_cells
    
    positions = []
    
    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                # Corner atom
                positions.append([epsilon + x * lattice_constant, epsilon + y * lattice_constant, epsilon + z * lattice_constant])
                # Body-centered atom
                positions.append([epsilon + (x + 0.5) * lattice_constant, epsilon + (y + 0.5) * lattice_constant, epsilon + (z + 0.5) * lattice_constant])
    
    # Convert positions to a numpy array
    positions = np.array(positions)
    
    # If more particles are generated than needed, trim the array
    if len(positions) > num_particles:
        positions = positions[:num_particles]
    
    return positions

# Define the dimensions of the box
# L =  [4.18, 8.36, 12.54, 16.72, 20.9, 25.08, 29.26, 33.44]

# # Number of particles
# num_particles = [2, 16, 54, 128, 250, 432, 686, 1024]

# # Generate BCC positions
# folder = 'input_files_bcc/'

# m_Na = 22.99
# m_Cl = 35.453
# header = "charge,mass,radius,x,y,z\n"

# for i in range(len(L)):
#     positions = generate_bcc_positions(L[i], num_particles[i])
#     filename = folder + 'input_coord' + str(num_particles[i]) + '.csv'
#     # Header for the CSV file
#     print('Density = ', 0.5 * num_particles[i] * (m_Cl + m_Na) / L[i]**3)
    
#     # Write coordinates to CSV file
#     with open(filename, "w", newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header.strip().split(','))
    
#         idx = 0
#         for pos in positions:
#             charge = (-1) ** idx  # Alternate the charge between 1 and -1
#             if charge > 0:
#                 mass = m_Na
#             else:
#                 mass = m_Cl
#             radius = 1
#             writer.writerow([charge, mass, radius, pos[0], pos[1], pos[2]])
#             idx += 1

#     print(f"CSV file '{filename}' has been generated.")