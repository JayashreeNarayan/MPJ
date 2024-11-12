import numpy as np
from input import grid_setting, a0, md_variables, output_settings
import math
from indices import GetDictTF
from scipy.interpolate import CubicSpline, LinearNDInterpolator, interpn, make_interp_spline

from profiling import profile

L = grid_setting.L
h = grid_setting.h
N = grid_setting.N
N_tot = grid_setting.N_tot
kBT = md_variables.kBT
potential_info = md_variables.potential
restart = output_settings.restart
input_restart_filename = grid_setting.input_restart_filename

# Particle class
class Particle:
    def __init__(self, charge, mass, radius, pos): # the particles can be anywhere in the grid, its molten salt - grid points is not the same as particle spots
        self.mass = mass            # mass of particle 
        self.pos = pos              # position of the particle in an array (3D)
        self.vel = np.zeros(3)      # velocity of the particle in an array (3D)
        self.charge = charge        # charge of the particle
        self.radius = radius        # radius of the particle
        self.neigh = np.zeros(8)    # 8 nearest neighbours of the particle
        self.force = np.zeros(3)    # force acting on the particle, electrostatic 
        self.force_notelec = np.zeros(3) # TF or LJ

        if potential_info == 'TF':
            self.r_cutoff = 0.5 * L # limit to where the force acts
            self.ComputeForceNotElec = self.ComputeTFForce
            #parameters
            self.B = 3.1546 * a0 # Mouhat 2013 
            self.dict = GetDictTF() # all the TF parameters
        
        elif potential_info == 'LJ':
            #parameters
            self.sigma = 3.00512 * 2 / a0   # given in Angs and converted in atomic units  #Ar = 3.400 / a0
            self.epsilon = 5.48 * 1e-4  #Hartree - corresponds to 0.0104 eV  #Ar = 3.82192993 * 1e-4

            self.r_cutoff = 2.5 * self.sigma
            self.ComputeForceNotElec = self.ComputeLJForce
            
            
    # find 8 nearest neighbours of the particle
    def NearestNeigh(self):
        indices = [math.floor(self.pos[i] / h)  for i in range(3)] # nearest indices from the particle's position -> i.e those that charge will be spread to
        neigh_indices = []        
        
        for i in range(indices[0], indices[0] + 2):
            for j in range(indices[1], indices[1] + 2):
                for k in range(indices[2], indices[2] + 2):
                    #  n_pos = dict_indices_CoordTon[tuple([i % N, j % N, k % N])] 
                     neigh_indices.append((i % N, j % N, k % N))
        
        self.neigh = np.array(neigh_indices).astype(int)


    # move the particle of a vector delta
    def Move(self, delta):
        self.pos = self.pos + delta
        
    # compute the force acting on the particle
    def ComputeForce(self, grid, prev): #IGNORE
        self.force = np.zeros(3)
 
        if prev == True: # this occurs only for the first step in verlet
            phi_v = np.copy(grid.phi_prev)
        else:
            phi_v = np.copy(grid.phi)

        #Qa = self.charge
        for i,j,k in self.neigh:
            # i,j,k = dict_indices_nToCoord[n]
            diff = self.pos - np.array([i,j,k]) * h #r_alpha - r_i
            self.force[0] -= phi_v[i,j,k] * self.charge * g_prime(diff[0]) * g(diff[1]) * g(diff[2]) 
            self.force[1] -= phi_v[i,j,k] * self.charge * g(diff[0]) * g_prime(diff[1]) * g(diff[2])
            self.force[2] -= phi_v[i,j,k] * self.charge * g(diff[0]) * g(diff[1]) * g_prime(diff[2]) 

        print(self.force)

    @profile
    def ComputeForce_FD(self, grid, prev): # defn of the force from notes poisson, computer force on the particle
        self.force = np.zeros(3)
       
        if prev == True:
            phi_v = grid.phi_prev 
        else:
            phi_v = grid.phi
        
        E_x = lambda i_p, j_p, k_p : (phi_v[(i_p - 1) % N,j_p, k_p] - phi_v[(i_p + 1) % N, j_p, k_p]) / (2 * h)
        E_y = lambda i_p, j_p, k_p : (phi_v[i_p, ((j_p - 1) % N), k_p ] - phi_v[i_p, ((j_p + 1) % N), k_p]) / (2 * h)
        E_z = lambda i_p, j_p, k_p : (phi_v[i_p, j_p, ((k_p - 1) % N)] - phi_v[i_p, j_p, ((k_p + 1) % N)]) / (2 * h)

        q_tot = 0
        for i,j,k in self.neigh:
            # i,j,k = dict_indices_nToCoord[n]
            self.force[0] = self.force[0] + grid.q[i,j,k] * E_x(i,j,k) # cumulative sum
            self.force[1] = self.force[1] + grid.q[i,j,k] * E_y(i,j,k) 
            self.force[2] = self.force[2] + grid.q[i,j,k] * E_z(i,j,k) 
            q_tot = q_tot + grid.q[i,j,k]
    
    
    def ComputeLJForcePair(self,particle):  # computes LJ force for couple of particles (self particle and a particle given in input
        r_diff = self.pos - particle.pos 
        r = BoxScale(r_diff)
        r_mag = BoxScaleDistance(r_diff)
        r_cap = r / r_mag
        
        if r_mag <= self.r_cutoff: 
            f_mag = 24 * self.epsilon / r_mag * (2 * (self.sigma/r_mag)**12 - (self.sigma/r_mag)**6)
        else:
            f_mag = 0
   
        return f_mag * r_cap
    
    def ComputeLJPotential(self,particle):  # potential between particles
        r_diff = self.pos - particle.pos 
        r_mag = BoxScaleDistance(r_diff)
        c_shift = -LJPotential(self.r_cutoff_LJ, self.epsilon, self.sigma)
        
        if r_mag <= self.r_cutoff_LJ: 
            V_mag = 4 * self.epsilon * ((self.sigma/r_mag)**12 - (self.sigma/r_mag)**6) + c_shift
        else:
            V_mag = 0
   
        return V_mag

    def ComputeLJForcePotentialPair(self, particle): # potential and force in the same function, between 2 particles
        r_diff = self.pos - particle.pos 
        r = BoxScale(r_diff)
        r_mag = BoxScaleDistance(r_diff)
        r_cap = r / r_mag
        c_shift = -LJPotential(self.r_cutoff, self.epsilon, self.sigma)
       
        if r_mag <= self.r_cutoff: 
            f_mag = 24 * self.epsilon / r_mag * (2 * (self.sigma/r_mag)**12 - (self.sigma/r_mag)**6)
            V_mag = 4 * self.epsilon * ((self.sigma/r_mag)**12 - (self.sigma/r_mag)**6) + c_shift
        else:
            f_mag = 0
            V_mag = 0
   
        return f_mag * r_cap, V_mag
    
    
    def ComputeLJForcePotential(self, particles): # total force on particle self.
        self.force_potential = np.zeros(3)

        for particle in particles:
            if(particle == self):
                continue
            else:
                force = force + self.ComputeLJForcePair(particle)

        self.force_notelec = force
    
    @profile
    def ComputeTFForcePotentialPair(self,particle):  
        r_diff = self.pos - particle.pos 
        # r = BoxScale(r_diff)
        # r_mag = BoxScaleDistance(r_diff)
        r, r_mag = BoxScaleDistance2(r_diff)
        r_cap = r / r_mag
        
        A, C, D, sigma_TF = self.dict[self.charge + particle.charge]
        V_shift = A * np.exp(self.B * (sigma_TF - self.r_cutoff)) - C / self.r_cutoff**6 - D / self.r_cutoff**8
        alpha = A * self.B * np.exp(self.B * (sigma_TF - self.r_cutoff)) - 6 * C / self.r_cutoff**7 - 8 * D / self.r_cutoff**9
        beta = - V_shift - alpha * self.r_cutoff

        #f_shift = self.B * A * np.exp(self.B * (sigma_TF - self.r_cutoff)) - 6 * C / self.r_cutoff**7 - 8 * D / self.r_cutoff**9 
        
        if r_mag <= self.r_cutoff: 
            f_mag = self.B * A * np.exp(self.B * (sigma_TF - r_mag)) - 6 * C / r_mag**7 - 8 * D / r_mag**9 - alpha
            V_mag = A * np.exp(self.B * (sigma_TF - r_mag)) - C / r_mag**6 - D / r_mag**8 + alpha * r_mag + beta #- V_shift
        else:
            f_mag = 0
            V_mag = 0.
            
        #print(f_mag * r_cap, V_mag)
        return f_mag * r_cap, V_mag
    

    def ComputeTFForce(self, particles): # total force acting on particle self
        force = np.zeros(3)

        for particle in particles:
            if(particle == self):
                continue
            else:
                force += self.ComputeTFForcePair(particle)
        #print('FORCE TF',self.force_TF)
        self.force_notelec = force

    
    def ComputeLJForce(self, particles): 
        force = np.zeros(3)

        for particle in particles:
            if(particle == self):
                continue
            else:
                force += self.ComputeLJForcePair(particle)
        #print('FORCE TF',self.force_TF)
        self.force_notelec = force

    
    def ComputeTFPotential(self, particles):
        pot = 0

        for particle in particles:
            if(particle == self):
                continue
            else:
                pot += self.ComputeTFPotentialPair(particle)

        return pot
       
    def ComputeTFForcePotential(self, particles):
        pot = 0
        force = np.zeros(3)

        for particle in particles:
            if(particle == self):
                continue
            else:
                pot += self.ComputeTFPotentialPair(particle)
                force += self.ComputeTFForcePair(particle)
                
        return force, pot

# distance with periodic boundary conditions
@profile
def BoxScaleDistance(diff): # returns a number - smallest distance between 2 neightbouring particles - to enforce PBC
    diff = diff - L * np.rint(diff / L)
    # distance = np.sqrt(np.dot(diff, diff))
    distance = np.linalg.norm(diff)
    return distance

@profile
def BoxScaleDistance2(diff): # returns a number - smallest distance between 2 neightbouring particles - to enforce PBC
    diff = diff - L * np.rint(diff / L)
    # distance = np.sqrt(np.dot(diff, diff))
    distance = np.linalg.norm(diff)
    return diff, distance

@profile
def BoxScale(diff): # returns a vector - smallest distance between 2 neightbouring particles - to enforce PBC
    diff = diff - L * np.rint(diff / L)
    return diff

# weight function as defined in the paper Im et al. (1998) - eqn 24
def g(x):
    x = x - L * np.rint(x / L)
    x = np.abs(x)
    
    if x < h:
        return 1 - x / h
    else:
        return 0

# derivative of the weight function as defined in the paper Im et al. (1998) - eqn 27
def g_prime(x):
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

# all possible neighbouring grid points
def i_vec(i):
    return  np.array([i - 2, i - 1, i, i + 1, i + 2])

def j_vec(j):
    return  np.array([j - 2, j - 1, j, j + 1, j + 2])

def k_vec(k):
    return  np.array([k - 2, k - 1, k, k + 1, k + 2])


