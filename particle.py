import numpy as np
from input import grid_setting, a0, md_variables, output_settings
import math
from indices import dict_indices_CoordTon, dict_indices_nToCoord, GetDictTF
from scipy.interpolate import CubicSpline, LinearNDInterpolator, interpn, make_interp_spline

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
    def __init__(self, charge, mass, radius, pos):
        self.mass = mass
        self.pos = pos              # position of the particle
        self.vel = np.zeros(3)                            # velocity of the particle
        self.charge = charge        # charge of the particle
        self.radius = radius        # radius of the particle
        self.neigh = np.zeros(8)    # 8 nearest neighbours of the particle
        self.force = np.zeros(3)    # force acting on the particle
        self.force_notelec = np.zeros(3) # TF or LJ

        if potential_info == 'TF':
            self.r_cutoff = 0.5 * L
            self.ComputeForceNotElec = self.ComputeTFForce
            #parameters
            self.B = 3.1546 * a0
            self.dict = GetDictTF()
        
        elif potential_info == 'LJ':
            #parameters
            self.sigma = 3.00512 * 2 / a0   # given in Angs and converted in atomic units  #Ar = 3.400 / a0
            self.epsilon = 5.48 * 1e-4  #Hartree - corresponds to 0.0104 eV  #Ar = 3.82192993 * 1e-4

            self.r_cutoff = 2.5 * self.sigma
            self.ComputeForceNotElec = self.ComputeLJForce
            
            
    # find 8 nearest neighbours of the particle
    def NearestNeigh(self):
        indices = [math.floor(self.pos[i] / h)  for i in range(3)]
        neigh_indices = []        
        
        for i in range(indices[0], indices[0] + 2):
            for j in range(indices[1], indices[1] + 2):
                for k in range(indices[2], indices[2] + 2):
                     n_pos = dict_indices_CoordTon[tuple([i % N, j % N, k % N])] 
                     neigh_indices.append(n_pos)
        
        self.neigh = np.array(neigh_indices).astype(int)


    # move the particle of a vector delta
    def Move(self, delta):
        self.pos = self.pos + delta
        
    # compute the force acting on the particle
    def ComputeForce(self, grid, prev):
        self.force = np.zeros(3)
 
        if prev == True:
            phi_v = np.copy(grid.phi_prev)
        else:
            phi_v = np.copy(grid.phi)

        #Qa = self.charge
        for n in self.neigh:
            i,j,k = dict_indices_nToCoord[n]
            diff = self.pos - np.array([i,j,k]) * h #r_alpha - r_i
            self.force[0] = self.force[0] - phi_v[n] * self.charge * g_prime(diff[0]) * g(diff[1]) * g(diff[2])
            self.force[1] = self.force[1] - phi_v[n] * self.charge * g(diff[0]) * g_prime(diff[1]) * g(diff[2])
            self.force[2] = self.force[2] - phi_v[n] * self.charge * g(diff[0]) * g(diff[1]) * g_prime(diff[2]) 

        print(self.force)

    '''
    def ComputeForce_CubicSpline(self, grid, prev):
        self.force = np.zeros(3)
      
        if prev == True:
            phi_v = grid.phi_prev
        else:
            phi_v = grid.phi

        #grad_phi_neigh = np.gradient(phi_neigh,h)
        for n in self.neigh:
            i,j,k = dict_indices_nToCoord[n]
            q_n = grid.q[n]
            # x component of the force
            i_vec = np.array([i - 2, i - 1, i, i + 1, i + 2])
            #i_vec = np.array([i - 1, i, i + 1])
            
            phi_x = np.array([phi_v[p % N + j * N + k * N * N] for p in i_vec])
            #phi_x = phi_v[i_vec % N + j * N + k * N * N]

            cs_x = CubicSpline(i_vec, phi_x)
            self.force[0] = self.force[0] + q_n * cs_x(i, 1)  
            
            # y component of the force
            j_vec = np.array([j - 2, j - 1, j, j + 1, j + 2])
            #j_vec = np.array([j - 1, j, j + 1])
            phi_y = np.array([phi_v[i + (p % N) * N + k * N * N] for p in j_vec])
            #phi_y = phi_v[i + (j_vec % N) * N + k * N * N] 

            cs_y = CubicSpline(j_vec, phi_y)
            self.force[1] = self.force[1] + q_n * cs_y(j, 1)

            # z component of the force
            k_vec = np.array([k - 2, k - 1, k, k + 1, k + 2])
            #k_vec = np.array([k - 1, k, k + 1])
            phi_z = np.array([phi_v[i + j * N + (p % N) * N * N] for p in k_vec])
            #phi_z = phi_v[i + j * N + (k_vec % N) * N * N] 

            cs_z = CubicSpline(k_vec, phi_z)
            self.force[2] = self.force[2] + q_n * cs_z(k, 1)
        
        #print('FORCE SPLINE',self.force,'\n') 

    def ComputeForce_ParticlePos(self, grid, prev):
        self.force = np.zeros(3)
      
        if prev == True:
            phi_v = grid.phi_prev
        else:
            phi_v = grid.phi
        
        phi_vec = np.array([phi_v[n] for n in self.neigh])

        i_vec = np.zeros(2)
        j_vec = np.zeros(2)
        k_vec = np.zeros(2)

        i_vec[0], j_vec[0], k_vec[0] = dict_indices_nToCoord[self.neigh[0]]
        i_vec[1] = i_vec[0] + 1
        j_vec[1] = j_vec[0] + 1
        k_vec[1] = k_vec[0] + 1

        x_vec = i_vec * h
        y_vec = j_vec * h
        z_vec = k_vec * h

        X, Y, Z = np.meshgrid(x_vec, y_vec, z_vec)
        #points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
        #interp_func = LinearNDInterpolator(points, phi_vec)
        #phi_pos = interp_func(self.pos)
        #force_pos = interp_func(self.pos,1)
        # Creazione dell'interpolatore spline cubico
        spline = make_interp_spline(points, phi_vec)

        # Ora puoi valutare la funzione interpolata in un punto arbitrario, ad esempio in (0.5, 0.5, 0.5)
        interp_value = spline(self.pos)

        # Puoi anche calcolare la derivata della funzione interpolata in un punto arbitrario
        spline_derivative = spline.derivative()

        # Ora puoi valutare la derivata in un punto arbitrario, ad esempio in (0.5, 0.5, 0.5)
        interp_derivative_value = spline_derivative(self.pos)

        print("Valore interpolato in (0.5, 0.5, 0.5):", interp_value)
        print("Derivata interpolata in (0.5, 0.5, 0.5):", interp_derivative_value)

        
        #print('FORCE SPLINE',self.force,'\n') 
    '''

    def ComputeForce_FD(self, grid, prev):
        self.force = np.zeros(3)
       
        if prev == True:
            phi_v = grid.phi_prev 
        else:
            phi_v = grid.phi
        
        E_x = lambda i_p, j_p, k_p : (phi_v[(i_p - 1) % N + j_p * N + k_p * N * N] - phi_v[(i_p + 1) % N + j_p * N + k_p * N * N]) / (2 * h)
        E_y = lambda i_p, j_p, k_p : (phi_v[i_p + ((j_p - 1) % N) * N + k_p * N * N] - phi_v[i_p + ((j_p + 1) % N) * N + k_p * N * N]) / (2 * h)
        E_z = lambda i_p, j_p, k_p : (phi_v[i_p + j_p * N + ((k_p - 1) % N) * N * N] - phi_v[i_p + j_p * N + ((k_p + 1) % N) * N * N]) / (2 * h)

        q_tot = 0
        for n in self.neigh:
            i,j,k = dict_indices_nToCoord[n]
            self.force[0] = self.force[0] + grid.q[n] * E_x(i,j,k) 
            self.force[1] = self.force[1] + grid.q[n] * E_y(i,j,k) 
            self.force[2] = self.force[2] + grid.q[n] * E_z(i,j,k) 
            q_tot = q_tot + grid.q[n]

            '''
            diff = self.pos - np.array([i,j,k]) * h 
            q_n = self.charge * g(diff[0]) * g(diff[1]) * g(diff[2]) 
            self.force[0] = self.force[0] + q_n * E_x(i,j,k) 
            self.force[1] = self.force[1] + q_n * E_y(i,j,k) 
            self.force[2] = self.force[2] + q_n * E_z(i,j,k) 
            '''
        #print(q_tot)
        #print('FORCE DIFF',self.force,'\n') 
    
    
    def ComputeLJForcePair(self,particle):  
        r_diff = self.pos - particle.pos 
        r = BoxScale(r_diff)
        r_mag = BoxScaleDistance(r_diff)
        r_cap = r / r_mag
        
        if r_mag <= self.r_cutoff: 
            f_mag = 24 * self.epsilon / r_mag * (2 * (self.sigma/r_mag)**12 - (self.sigma/r_mag)**6)
        else:
            f_mag = 0
   
        return f_mag * r_cap
    
    def ComputeLJPotential(self,particle):  
        r_diff = self.pos - particle.pos 
        r_mag = BoxScaleDistance(r_diff)
        c_shift = -LJPotential(self.r_cutoff_LJ, self.epsilon, self.sigma)
        
        if r_mag <= self.r_cutoff_LJ: 
            V_mag = 4 * self.epsilon * ((self.sigma/r_mag)**12 - (self.sigma/r_mag)**6) + c_shift
        else:
            V_mag = 0
   
        return V_mag

    def ComputeLJForcePotentialPair(self, particle):
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
    
    
    def ComputeLJForcePotential(self, particles):
        self.force_potential = np.zeros(3)

        for particle in particles:
            if(particle == self):
                continue
            else:
                force = force + self.ComputeLJForcePair(particle)

        self.force_notelec = force
    
    
    def ComputeTFForcePair(self,particle):  
        r_diff = self.pos - particle.pos 
        r = BoxScale(r_diff)
        r_mag = BoxScaleDistance(r_diff)
        r_cap = r / r_mag
        
        A, C, D, sigma_TF = self.dict[self.charge + particle.charge]
        f_shift = self.B * A * np.exp(self.B * (sigma_TF - self.r_cutoff)) - 6 * C / self.r_cutoff**7 - 8 * D / self.r_cutoff**9 

        if r_mag <= self.r_cutoff: 
        #già incluso with linked cell, prendo solo quelli da calcolare
            f_mag = self.B * A * np.exp(self.B * (sigma_TF - r_mag)) - 6 * C / r_mag**7 - 8 * D / r_mag**9 - f_shift
        else:
        #    print('Hola')
            f_mag = 0
   
        return f_mag * r_cap
    
    
    def ComputeTFPotentialPair(self,particle):  
        r_diff = self.pos - particle.pos 
        r_mag = BoxScaleDistance(r_diff)

        A, C, D, sigma_TF = self.dict[self.charge + particle.charge]
        V_shift = A * np.exp(self.B * (sigma_TF - self.r_cutoff)) - C / self.r_cutoff**6 - D / self.r_cutoff**8

        if r_mag <= self.r_cutoff: 
            V_mag = A * np.exp(self.B * (sigma_TF - r_mag)) - C / r_mag**6 - D / r_mag**8 #- V_shift
        else:
            V_mag = 0
   
        return V_mag
    

    def ComputeTFForcePotentialPair(self,particle):  
        r_diff = self.pos - particle.pos 
        r = BoxScale(r_diff)
        r_mag = BoxScaleDistance(r_diff)
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
    

    def ComputeTFForce(self, particles):
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
def BoxScaleDistance(diff):
    diff = diff - L * np.rint(diff / L)
    distance = np.sqrt(np.dot(diff, diff))
    return distance


def BoxScale(diff):
    diff = diff - L * np.rint(diff / L)
    return diff

# weight function as defined in the paper Im et al. (1998)
def g(x):
    x = x - L * np.rint(x / L)
    x = np.abs(x)
    
    if x < h:
        return 1 - x / h
    else:
        return 0

# derivative of the weight function as defined in the paper Im et al. (1998)
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


def i_vec(i):
    return  np.array([i - 2, i - 1, i, i + 1, i + 2])

def j_vec(j):
    return  np.array([j - 2, j - 1, j, j + 1, j + 2])

def k_vec(k):
    return  np.array([k - 2, k - 1, k, k + 1, k + 2])


