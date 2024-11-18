import math
import numpy as np
from .constants import a0
from .indices import GetDictTF
from .profiling import profile


class Particles:
    def __init__(self, grid, md_variables, charges, masses, positions):
        self.grid = grid
        self.N_p = grid.N_p
        
        # Particle properties as NumPy arrays
        self.masses = np.array(masses)        # Shape: (N_p,)
        self.pos = np.array(positions) # Shape: (N_p, 3)
        self.vel = np.zeros((self.N_p, 3), dtype=float) # Shape: (N_p, 3)
        self.charges = np.array(charges)     # Shape: (N_p,)
        self.forces = np.zeros((self.N_p, 3), dtype=float) # Electrostatic forces
        self.forces_notelec = np.zeros((self.N_p, 3), dtype=float) # Non-electrostatic forces
        self.potential_info = md_variables.potential

        # Additional attributes based on grid potential
        L = grid.L
        if self.potential_info == 'TF':
            self.r_cutoff = 0.5 * L  # Limit to where the force acts
            self.B = 3.1546 * a0 # Parameter for TF potential
            self.tf_params = GetDictTF() # # Dictionary of TF parameters
            self.ComputeForceNotElec = self.ComputeTFForces
            
        elif self.potential_info == 'LJ':
            self.sigma = 3.00512 * 2 / a0 # Lennard-Jones sigma
            self.epsilon = 5.48 * 1e-4         # Lennard-Jones epsilon (Hartree)
            self.r_cutoff = 2.5 * self.sigma
            self.ComputeForceNotElec = self.ComputeLJForce

        # Pre-allocate for nearest neighbor indices
        self.neighbors = np.empty((self.N_p, 8, 3), dtype=int) # Shape: (N_p, 8, 3)


    def NearestNeighbors(self):
        N = self.grid.N
        h = self.grid.h

        # Compute indices for all particles
        indices = np.floor(self.pos / h).astype(int) # Shape: (N_p, 3)

        for n in range(self.N_p):
            neigh_indices = []
            for i in range(indices[n][0], indices[n][0] + 2):
                for j in range(indices[n][1], indices[n][1] + 2):
                    for k in range(indices[n][2], indices[n][2] + 2):
                        neigh_indices.append((i % N, j % N, k % N))

            self.neighbors[n] = neigh_indices
            
    # Currently using this one
    @profile 
    def ComputeForce_FD(self, prev):
        h = self.grid.h

        # Choose the appropriate potential
        phi_v = self.grid.phi_prev if prev else self.grid.phi

        # Precompute electric field components (central difference approximation)
        E_x = (np.roll(phi_v, -1, axis=0) - np.roll(phi_v, 1, axis=0)) / (2 * h)
        E_y = (np.roll(phi_v, -1, axis=1) - np.roll(phi_v, 1, axis=1)) / (2 * h)
        E_z = (np.roll(phi_v, -1, axis=2) - np.roll(phi_v, 1, axis=2)) / (2 * h)

        # Electric field at neighbor points for all particles (Shape: (n_particles, 8, 3))
        neighbors = self.neighbors  # Shape: (n_particles, 8, 3)
        q_neighbors = self.grid.q[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]  # Shape: (n_particles, 8)

        E_neighbors = np.stack([
            E_x[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]],
            E_y[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]],
            E_z[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]],
        ], axis=-1)  # Shape: (n_particles, 8, 3)

        # Compute the forces (sum contributions from 8 neighbors)
        self.forces = -np.sum(q_neighbors[:, :, np.newaxis] * E_neighbors, axis=1)  # Shape: (n_particles, 3)

        # Compute total charge contribution (optional)
        self.grid.q_tot = np.sum(q_neighbors)  # Scalar
    

    def ComputeTFForces(self):
        # Get all pairwise differences
        r_diff = self.pos[:, np.newaxis, :] - self.pos[np.newaxis, :, :]  # Shape: (N_p, N_p, 3)

        # Compute pairwise distances and unit vectors
        r_diff = BoxScale(r_diff, self.grid.L)  # Apply periodic boundary conditions
        r_mag = np.linalg.norm(r_diff, axis=2)  # Shape: (N_p, N_p)
        np.fill_diagonal(r_mag, np.inf)  # Avoid self-interaction by setting diagonal to infinity

        r_cap = np.divide(r_diff, r_mag[:, :, np.newaxis], where=r_mag[:, :, np.newaxis] != 0)  # Avoid division by zero

        # Compute pairwise TF parameters
        charges_sum = self.charges[:, np.newaxis] + self.charges[np.newaxis, :]  # Shape: (N_p, N_p)
        A = np.vectorize(lambda q: self.tf_params[q][0])(charges_sum)
        C = np.vectorize(lambda q: self.tf_params[q][1])(charges_sum)
        D = np.vectorize(lambda q: self.tf_params[q][2])(charges_sum)
        sigma_TF = np.vectorize(lambda q: self.tf_params[q][3])(charges_sum)

        # Apply cutoff mask
        within_cutoff = r_mag <= self.r_cutoff

        V_shift = A * np.exp(self.B * (sigma_TF - self.r_cutoff)) - C / self.r_cutoff**6 - D / self.r_cutoff**8
        alpha = A * self.B * np.exp(self.B * (sigma_TF - self.r_cutoff)) - 6 * C / self.r_cutoff**7 - 8 * D / self.r_cutoff**9
        beta = - V_shift - alpha * self.r_cutoff
        
        # Compute force magnitudes and potentials only for particles within the cutoff
        f_mag = np.where(within_cutoff, self.B * A * np.exp(self.B * (sigma_TF - r_mag)) - 6 * C / r_mag**7 - 8 * D / r_mag**9 - alpha, 0)
        V_mag = np.where(within_cutoff, A * np.exp(self.B * (sigma_TF - r_mag)) - C / r_mag**6 - D / r_mag**8 + alpha * r_mag + beta, 0) #- V_shift

        # Compute pairwise forces, now with the shift applied to the force magnitudes
        pairwise_forces = f_mag[:, :, np.newaxis] * r_cap  # Shape: (N_p, N_p, 3)

        # Sum forces to get the net force on each particle
        net_forces = np.nansum(pairwise_forces, axis=1)  # Sum forces acting on each particle (ignore NaN values)

        # Update the instance variables
        self.forces_notelec = net_forces  # Store net forces in the instance variable
        potential_energy = np.nansum(V_mag) / 2  # Avoid double-counting for potential energy

        self.grid.potential_notelec = potential_energy  # Store the potential energy


    def ComputeForce(self, grid, prev):
        L = grid.L
        h = grid.h

        # Select the correct potential
        phi_v = grid.phi_prev if prev else grid.phi

        # Convert neighbor indices to coordinates
        neigh_coords = np.array(self.neigh) * h  # Shape: (num_neighbors, 3)

        # Compute vector differences: r_alpha - r_i for all neighbors
        diffs = self.pos - neigh_coords  # Shape: (num_neighbors, 3)

        # Compute g and g_prime for all components
        g_x = g(diffs[:, 0], L, h)
        g_y = g(diffs[:, 1], L, h)
        g_z = g(diffs[:, 2], L, h)

        g_prime_x = g_prime(diffs[:, 0], L, h)
        g_prime_y = g_prime(diffs[:, 1], L, h)
        g_prime_z = g_prime(diffs[:, 2], L, h)

        # Extract the phi values for all neighbors
        phi_values = phi_v[tuple(np.array(self.neigh).T)]  # Shape: (num_neighbors,)

        # Compute force contributions for each component
        self.force = -self.charge * np.array([
            np.sum(phi_values * g_prime_x * g_y * g_z),
            np.sum(phi_values * g_x * g_prime_y * g_z),
            np.sum(phi_values * g_x * g_y * g_prime_z)
        ])


  
# distance with periodic boundary conditions
@profile
# returns a number - smallest distance between 2 neightbouring particles - to enforce PBC
def BoxScaleDistance(diff, L): 
    diff = diff - L * np.rint(diff / L)
    distance = np.linalg.norm(diff)
    return distance
    
@profile
# returns a number - smallest distance between 2 neightbouring particles - to enforce PBC
def BoxScaleDistance2(diff, L): 
    diff = diff - L * np.rint(diff / L)
    distance = np.linalg.norm(diff)
    return diff, distance

@profile
# returns a vector - smallest distance between 2 neightbouring particles - to enforce PBC
def BoxScale(diff, L): 
    diff = diff - L * np.rint(diff / L)
    return diff

# weight function as defined in the paper Im et al. (1998) - eqn 24
def g(x, L, h):
    x = x - L * np.rint(x / L)
    x = np.abs(x)
    
    # Use vectorized operations with NumPy
    result = np.where(x < h, 1 - x / h, 0)
    return result


# derivative of the weight function as defined in the paper Im et al. (1998) - eqn 27
def g_prime(x, L, h):
    x = x - L * np.rint(x / L)
    if x < 0:
        return 1 / h
    elif x == 0:
        return 0
    else:
        return - 1 / h
    
def LJPotential(r, epsilon, sigma):  
        V_mag = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        return V_mag

