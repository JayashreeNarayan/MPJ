import numpy as np
from input import grid_setting, a0, md_variables
import math
from indices import dict_indices_CoordTon, dict_indices_nToCoord, GetDictTF
from scipy.interpolate import CubicSpline, LinearNDInterpolator, interpn, make_interp_spline

L = grid_setting.L
h = grid_setting.h
N = grid_setting.N
N_tot = grid_setting.N_tot
kBT = md_variables.kBT
potential_info = md_variables.potential

# Particle class
class Particle:
    def __init__(self, charge, mass, radius, pos):
        self.mass = mass
        self.pos = pos              # position of the particle
        self.vel = np.array([np.random.normal(loc = 0.0, scale = np.sqrt(kBT / self.mass)) for i in range(3)])                                  # velocity of the particle
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
            self.r_cutoff = 2.5 * self.sigma
            self.ComputeForceNotElec = self.ComputeLJForce
            
            #parameters
            self.sigma = 3.00512 * 2 / a0   # given in Angs and converted in atomic units  #Ar = 3.400 / a0
            self.epsilon = 5.48 * 1e-4  #Hartree - corresponds to 0.0104 eV  #Ar = 3.82192993 * 1e-4
            
        
        #self.n_neigh =  np.ceil(radius / h) + 1
        #self.cube_indices = np.arange(-self.n_neigh, self.n_neigh + 1, 1, dtype=int)

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
        
        for n in self.neigh[:4]:
            i,j,k = dict_indices_nToCoord[n]
            diff = self.pos - np.array([i,j,k]) * h #r_alpha - r_i
            self.force[0] = self.force[0] - phi_v[n] * self.charge * g_prime(diff[0]) * g(diff[1]) * g(diff[2])
            self.force[1] = self.force[1] - phi_v[n] * self.charge * g(diff[0]) * g_prime(diff[1]) * g(diff[2])
            self.force[2] = self.force[2] - phi_v[n] * self.charge * g(diff[0]) * g(diff[1]) * g_prime(diff[2]) 

        '''
        for n in self.neigh:
            coord = dict_indices_nToCoord[n]
            diff = self.pos - coord * h #r_alpha - r_i
            self.force = self.force - phi_v[n] * self.charge * grad_g(diff[0],diff[1], diff[2])
        '''
        #print('FORCE BENOIT',self.force)


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


    def ComputeForce_FD(self, grid, prev):
        self.force = np.zeros(3)
      
        if prev == True:
            phi_v = grid.phi_prev
        else:
            phi_v = grid.phi

        E_x = lambda i_p, j_p, k_p : (phi_v[(i_p - 1) % N + j_p * N + k_p * N * N] - phi_v[(i_p + 1) % N + j_p * N + k_p * N * N]) / (2 * h)
        E_y = lambda i_p, j_p, k_p : (phi_v[i_p + ((j_p - 1) % N) * N + k_p * N * N] - phi_v[i_p + ((j_p + 1) % N) * N + k_p * N * N]) / (2 * h)
        E_z = lambda i_p, j_p, k_p : (phi_v[i_p + j_p * N + ((k_p - 1) % N) * N * N] - phi_v[i_p + j_p * N + ((k_p + 1) % N) * N * N]) / (2 * h)

        for n in self.neigh:
            i,j,k = dict_indices_nToCoord[n]
            self.force[0] = self.force[0] + grid.q[n] * E_x(i,j,k) 
            self.force[1] = self.force[1] + grid.q[n] * E_y(i,j,k) 
            self.force[2] = self.force[2] + grid.q[n] * E_z(i,j,k) 
        
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
    
    '''
    def ComputeLJPotential(self,particle):  
        r_diff = self.pos - particle.pos 
        r_mag = BoxScaleDistance(r_diff)
        c_shift = -LJPotential(self.r_cutoff_LJ, self.epsilon, self.sigma)
        
        if r_mag <= self.r_cutoff_LJ: 
            V_mag = 4 * self.epsilon * ((self.sigma/r_mag)**12 - (self.sigma/r_mag)**6) + c_shift
        else:
            V_mag = 0
   
        return V_mag
    '''

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

        if r_mag <= self.r_cutoff: 
        #già incluso with linked cell, prendo solo quelli da calcolare
            f_mag = self.B * A * np.exp(self.B * (sigma_TF - r_mag)) - 6 * C / r_mag**7 - 8 * D / r_mag**9 
        else:
        #    print('Hola')
            f_mag = 0
   
        return f_mag * r_cap
    
    
    def ComputeTFPotentialPair(self,particle):  
        r_diff = self.pos - particle.pos 
        r_mag = BoxScaleDistance(r_diff)
        '''
        if (self.charge == 1 and particle.charge == -1) or (self.charge == -1 and particle.charge == 1): # Na - Cl
            A = 7.7527 * 1e-3 #Hartree = 20.3548 kJ/mol 
            C = 0.2569 / a0**6 # = 674.4798 kj/mol * Ang^6 
            D = 0.3188 / a0**8 # = 837.0777 kj/mol * Ang^8 
            sigma_TF = 2.755 / a0
   
        elif self.charge == 1 and particle.charge == 1: # Na - Na
            A = 9.6909 * 1e-3 #Hartree = 25.4435 kJ/mol 
            C = 0.0385 / a0**6 # = 101.1719 kj/mol * Ang^6 
            D = 0.0183 / a0**8 # = 48.1771 kj/mol * Ang^8 
            sigma_TF = 2.340 / a0 # = 2.340 angs
        
        elif self.charge == -1 and particle.charge == -1: #Cl - Cl
            A = 5.8145 * 1e-3 #Hartree = 15.2661 kJ/mol 
            C = 2.6607 / a0**6 # = 6985.6841 kj/mol * Ang^6 
            D = 5.3443 / a0**8 # = 14,031.5897 kj/mol * Ang^8 
            sigma_TF = 3.170 / a0 # = 3.170 angs
        '''
        A, C, D, sigma_TF = self.dict[self.charge + particle.charge]

        if r_mag <= self.r_cutoff: 
            V_mag = A * np.exp(self.B * (sigma_TF - r_mag)) - C / r_mag**6 - D / r_mag**8
        else:
            V_mag = 0
   
        return V_mag
    
    def ComputeTFForcePotentialPair(self,particle):  
        r_diff = self.pos - particle.pos 
        r = BoxScale(r_diff)
        r_mag = BoxScaleDistance(r_diff)
        r_cap = r / r_mag
        
        '''
        if (self.charge == 1 and particle.charge == -1) or (self.charge == -1 and particle.charge == 1): # Na - Cl
            A = 7.7527 * 1e-3 #Hartree = 20.3548 kJ/mol 
            C = 0.2569 / a0**6 # = 674.4798 kj/mol * Ang^6 
            D = 0.3188 / a0**8 # = 837.0777 kj/mol * Ang^8 
            sigma_TF = 2.755 / a0
            
        elif self.charge == 1 and particle.charge == 1: # Na - Na
            A = 9.6909 * 1e-3 #Hartree = 25.4435 kJ/mol 
            C = 0.0385 / a0**6 # = 101.1719 kj/mol * Ang^6 
            D = 0.0183 / a0**8 # = 48.1771 kj/mol * Ang^8 
            sigma_TF = 2.340 / a0 # 2.340 angs

        elif self.charge == -1 and particle.charge == -1: #Cl - Cl
            A = 5.8145 * 1e-3 #Hartree = 15.2661 kJ/mol 
            C = 2.6607 / a0**6 # = 6985.6841 kj/mol * Ang^6 
            D = 5.3443 / a0**8 # = 14,031.5897 kj/mol * Ang^8 
            sigma_TF = 3.170 / a0 # 3.170 angs
        '''

        A, C, D, sigma_TF = self.dict[self.charge + particle.charge]

        if r_mag <= self.r_cutoff: 
            #già incluso with linked cell, prendo solo quelli da calcolare
            f_mag = self.B * A * np.exp(self.B * (sigma_TF - r_mag)) - 6 * C / r_mag**7 - 8 * D / r_mag**9 
            V_mag = A * np.exp(self.B * (sigma_TF - r_mag)) - C / r_mag**6 - D / r_mag**8
        else:
            f_mag = 0
            V_mag = 0
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
    '''  
        # Coulomb force
        for n in self.neigh:
            for m in grid.particles[id].neigh:
                coord_n = dict_indices_nToCoord[n]
                coord_m = dict_indices_nToCoord[m]
                diff = (coord_n - coord_m) * h #r_alpha - r_i
                for d in diff:
                    d = d - L * np.round(d / L)
                dist = np.sqrt(np.dot(diff,diff))
                if dist != 0:
                    force = force + grid.q[n] * grid.q[m] * diff / (dist**3)

        #print(force[0], self.force[0])
        print(force)

    def ComputeInitialNeighField(self):
        # find the closest grid point to the center of the solute
        i, j, k = np.ceil(self.pos / h).astype(int)
       
       # store the values of the neighbours directly as the n indices
        n_list = np.zeros(len(self.cube_indices)**3)
        counter = 0

        # sum over the indices of the cube, computed starting from the number of neighbours per side that I want to include
        for l in self.cube_indices:
            for m in self.cube_indices:
                for p in self.cube_indices:
                    coord = tuple([ (i + l + N) % N, (j + m + N) % N, (k + p + N) % N])
                    n_list[counter] = dict_indices_CoordTon[coord]
                    counter = counter + 1
        
        return n_list
    '''

    
''' 
    def ComputeNeighField(self, old_list):
        i, j, k = np.ceil(self.pos / h).astype(int)
        indices = np.arange(-self.n_neigh, self.n_neigh + 1, 1, dtype=int)
        n_list = np.zeros(len(indices)**3)
        counter = 0

        for l in indices:
            for m in indices:
                for p in indices:
                    coord = tuple([ (i + l + N) % N, (j + m + N) % N, (k + p + N) % N])
                    n_list[counter] = dict_indices_CoordTon[coord]
                    counter = counter + 1
      
        n_all = np.unique(np.concatenate((old_list, n_list)))
        
        return n_all

        
# sorting algorithm
def BubbleSort(arr, arrLead):
    n = len(arrLead)
    swapped = False
    
    for i in range(n-1):
        for j in range(0, n-i-1):
 
            if arrLead[j] > arrLead[j + 1]:
                swapped = True
                arrLead[j], arrLead[j + 1] = arrLead[j + 1], arrLead[j]
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
         
        if not swapped:
            return        

# translate indices (i,j,k) to an index n = i + j * N + k * N * N
def nFromIndices(indices):
    i = indices[0] % N
    j = indices[1] % N
    k = indices[2] % N
    n = i + j * N + k * N * N
    return n


# modulus function
def NearbyInt(number):
    if np.fabs(number) >= 0.5:
        return 1.
    elif np.fabs(number) < 0.5:
        return 0.
    else:
        print("Error")



def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])

def NearbyInt(x):
    if np.size(x) == 1:
       return int(proper_round(x))
    else:
        num = []
        for i in x:
           num.append(int(proper_round(i)))
        return np.array(num)
'''

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
    #sgn_x = np.sign(x)
    #print('NEW X:')
    #print(x)
    x = x - L * np.rint(x / L)
    #print(x)
    #x = sgn_x * np.abs(x)
    
    if x < 0 and x > -h: 
        return 1 / h
    elif x == 0:
        return 0
    elif x > 0 and x < h:
        return - 1 / h
    else:
        return 0


    

def grad_g(x, y, z):
    # Apply periodic boundary conditions
    #x = x - L * np.rint(x / L)
    #y = y - L * np.rint(y / L)
    #z = z - L * np.rint(z / L)
    x = x / L
    x = x - np.rint(x)
    x = x * L
    
    y = y / L
    y = y - np.rint(y)
    y = y * L

    z = z / L
    z = z - np.rint(z)
    z = z * L
    
    #y = y % L
    #z = z % L
    
    # Partial derivative with respect to x
    dg_dx = 0
    if x > 0:
        dg_dx = -(1 - abs(y) / h) * (1 - abs(z) / h) / h
    elif x < 0:
        dg_dx = (1 - abs(y) / h) * (1 - abs(z) / h) / h

    # Partial derivative with respect to y
    dg_dy = 0
    if y > 0:
        dg_dy = -(1 - abs(x) / h) * (1 - abs(z) / h) / h
    elif y < 0:
        dg_dy = (1 - abs(x) / h) * (1 - abs(z) / h) / h

    # Partial derivative with respect to z
    dg_dz = 0
    if z > 0:
        dg_dz = -(1 - abs(x) / h) * (1 - abs(y) / h) / h
    elif z < 0:
        dg_dz = (1 - abs(x) / h) * (1 - abs(y) / h) / h

    return np.array((dg_dx, dg_dy, dg_dz))

def LJPotential(r, epsilon, sigma):  
        V_mag = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
        return V_mag

def i_vec(i):
    return  np.array([i - 2, i - 1, i, i + 1, i + 2])

def j_vec(j):
    return  np.array([j - 2, j - 1, j, j + 1, j + 2])

def k_vec(k):
    return  np.array([k - 2, k - 1, k, k + 1, k + 2])