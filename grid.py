from input import grid_setting, a0, md_variables
from particle import Particle, BoxScaleDistance, g, BoxScale
import numpy as np
from indices import dict_indices_nToCoord
import time 
import pandas as pd
from output_md import file_output_energy
from linkedcell import LinkedCell

# get input variables from input file
N = grid_setting.N
N_tot = grid_setting.N_tot
h = grid_setting.h
L = grid_setting.L
k_bar = grid_setting.k_bar
k_b = grid_setting.k_b
eps = grid_setting.eps
input_filename = grid_setting.input_filename
dt = md_variables.dt
potential_info = md_variables.potential
elec = md_variables.elec
not_elec = md_variables.not_elec

amu_to_kg = 1.66054 * 1e-27 
m_e = 9.1093837 * 1e-31 #kg
conv_mass = amu_to_kg / m_e
offset_update = np.array([[h/2, 0, 0], [h/2, 0, 0], [0, h/2, 0], [0, h/2, 0], [0, 0, h/2], [0, 0, h/2], [0, 0, 0]])
# grid class to represent the grid and the fields operating on it

class Grid:
    def __init__(self):
        df = pd.read_csv(input_filename)
        #print(df.head())
        self.N_p = grid_setting.N_p
        self.particles = []
        for i in range(self.N_p):
            self.particles.append(Particle(df['charge'][i],           #charge given in electronic charge units
                                           df['mass'][i] * conv_mass, #mass given in amu and converted in au
                                           (df['radius'][i] / a0),      #radius given in Angs and converted to au
                                           (np.array([df['x'][i], df['y'][i], df['z'][i]])) / a0))                                  
        
        self.q = np.array(np.zeros(N_tot))            # charge vector - q for every grid point
        self.phi = np.array(np.zeros(N_tot))          # electrostatic field updated with MaZe
        self.phi_prev = np.array(np.zeros(N_tot))     # electrostatic field for step t - 1 Verlet
        self.indices7 = np.zeros((N_tot, 7))
        self.linked_cell = None
        self.energy = 0
        self.potential_notelec = 0
        
        if potential_info == 'TF':
            self.ComputeForceNotElecLC = self.ComputeForcesTFLinkedcell
            self.ComputeForceNotElecBasic = self.ComputeForcesTFBasic
        elif potential_info == 'LJ':
            self.ComputeForceNotElecLC = self.ComputeForcesLJLinkedcell
            self.ComputeForceNotElecBasic = self.ComputeForcesLJBasic
      
    def LinkedCellInit(self,cutoff):
        self.linked_cell = LinkedCell(cutoff, L, self.N_p)
        self.linked_cell.update_lists(self.particles)
    

    def ComputeForcesLJBasic(self):
        pe = 0

        for p1 in range(self.N_p):
            for p2 in range(p1+1, self.N_p):
                pair_force, pair_potential = self.particles[p1].ComputeLJForcePotentialPair(self.particles[p2]) 
                self.particles[p1].force_notelec += pair_force
                self.particles[p2].force_notelec -= pair_force
                            
                pe += pair_potential
        self.potential_notelec = pe

    # LINKED-CELL METHOD
    def ComputeForcesLJLinkedcell(self):
        for p in self.particles:   
            p.force_notelec = np.zeros(3)
            
        pe = 0
        side_cells = self.linked_cell.side_cell_num
        for IX in range(side_cells):
            for IY in range(side_cells):
                for IZ in range(side_cells):
                    central_cell_particles, neighbor_cells_particles = self.linked_cell.interacting_particles(IX, IY, IZ)
                    num_particles_central = len(central_cell_particles)
                    
                    for p1 in range(num_particles_central):
                        #Interaction between particles in central cell
                        for p2 in range(p1+1,num_particles_central):
                            pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeLJForcePotentialPair(self.particles[central_cell_particles[p2]]) 
                            self.particles[central_cell_particles[p1]].force_notelec += pair_force
                            self.particles[central_cell_particles[p2]].force_notelec -= pair_force
                            
                            pe += pair_potential
                    
                        #Interaction between particles in central cell and neighbors
                        if side_cells > 2:
                            for pn in neighbor_cells_particles:
                                pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeLJForcePotentialPair(self.particles[pn]) 
                                self.particles[central_cell_particles[p1]].force_notelec += pair_force
                                self.particles[pn].force_notelec -= pair_force
                                pe += pair_potential

        self.potential_notelec = pe
    

    def ComputeForcesTFBasic(self):
        for p in self.particles:
            p.force_notelec = np.zeros(3)
        pe = 0
        
        for p1 in range(self.N_p):
            for p2 in range(p1+1, self.N_p):
                pair_force, pair_potential = self.particles[p1].ComputeTFForcePotentialPair(self.particles[p2]) 
                self.particles[p1].force_notelec += pair_force
                self.particles[p2].force_notelec -= pair_force
                            
                pe += pair_potential
                
        self.potential_notelec = pe


    def ComputeForcesTFLinkedcell(self):
        for p in self.particles:
            p.force_notelec = np.zeros(3)

        pe = 0
        side_cells = self.linked_cell.side_cell_num
        
        for IX in range(side_cells):
            for IY in range(side_cells):
                for IZ in range(side_cells):
                    central_cell_particles, neighbor_cells_particles = self.linked_cell.interacting_particles(IX, IY, IZ)
                    num_particles_central = len(central_cell_particles)
                    
                    for p1 in range(num_particles_central):
                        #Interaction between particles in central cell
                        for p2 in range(p1+1,num_particles_central):
                            pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeTFForcePotentialPair(self.particles[central_cell_particles[p2]]) 
                            self.particles[central_cell_particles[p1]].force_notelec += pair_force
                            self.particles[central_cell_particles[p2]].force_notelec -= pair_force
                            
                            pe += pair_potential
                    
                        #Interaction between particles in central cell and neighbors
                        if side_cells > 2:
                            for pn in neighbor_cells_particles:
                                #print(central_cell_particles[p1],pn)
                                pair_force, pair_potential = self.particles[central_cell_particles[p1]].ComputeTFForcePotentialPair(self.particles[pn]) 
                                self.particles[central_cell_particles[p1]].force_notelec += pair_force
                                self.particles[pn].force_notelec -= pair_force
                                pe += pair_potential

        self.potential_notelec = pe


    # update charges with a weight function that spreads it on the grid
    def SetCharges(self):
        self.q = np.array(np.zeros(N_tot))

        q_tot = 0
        q_tot_expected = 0
        
        for particle in self.particles:
            q_tot_expected = q_tot_expected + particle.charge

            for n in particle.neigh:
                coord = dict_indices_nToCoord[n]
                diff = particle.pos - coord * h
                #diff = BoxScale(diff)
                self.q[n] += particle.charge * g(diff[0]) * g(diff[1]) * g(diff[2])
                #print(self.q[n], g(diff[0]), g(diff[1]), g(diff[2]))
                q_tot = q_tot + self.q[n]

        if q_tot + 1e-6 < q_tot_expected:
            print('Error: change initial position, charge is not preserved - q_tot =', q_tot) 


    def Energy(self, iter,prev=False):
        
        if prev == False:
            phi_v = self.phi
        else:
            phi_v = self.phi_prev

        # electrostatic potential
        if elec:
            potential = 0
            pot1 = 0

            for p in self.particles:
                for n in p.neigh:
                    coord = dict_indices_nToCoord[n]
                    diff = p.pos - coord * h
                    q_n = p.charge * g(diff[0]) * g(diff[1]) * g(diff[2])
                    #potential = potential + p.charge * phi_v[n] * g(diff[0])* g(diff[1])* g(diff[2])
                    pot1 = pot1 + 0.5 * q_n * phi_v[n]
                    potential = potential + 0.5 * self.q[n] * phi_v[n]
        
        #pot = 0
        #for n in range(N_tot):
        #    pot = pot + 0.5 * self.q[n] * phi_v[n]
        #print(pot1, potential)
        '''    
        # LJ potential
        potential_LJ = 0
            
        for p1 in self.particles:
            for p2 in self.particles:
                if p2 == p1:
                    continue
                else:
                    potential_LJ = potential_LJ + p1.ComputeLJPotential(p2)
        potential_LJ = potential_LJ * 0.5
        '''
        '''
        # TF potential
        potential_TF = 0
            
        for p1 in self.particles:
            for p2 in self.particles:
                if p2 == p1:
                    continue
                else:
                    potential_TF = potential_TF + p1.ComputeTFPotential(p2)
        potential_TF = potential_TF * 0.5
        '''
        # kinetic E
        kinetic = 0
            
        for p in self.particles:
            kinetic = kinetic + 0.5 * p.mass * np.dot(p.vel, p.vel)
        #print('potential = ', potential, ' TF = ', self.potential_notelec)
        if elec and not_elec:
            pot_tot = self.potential_notelec + potential
        elif elec and not_elec == False:
            pot_tot = potential
        elif not_elec and elec == False:
            pot_tot = self.potential_notelec


        self.energy = kinetic + pot_tot #+ potential #+ potential_LJ
        file_output_energy.write(str(iter * dt) + ',' +  str(self.energy) + ',' +  str(kinetic) + ',' + str(pot_tot) + '\n')#+ potential) + '\n')

    # compute the electrostatic potential considering the charges spread on the grid
    #def ComputePotentialn(self, m_point):
    #    potential = 0
    #    coord_m = nToGridCoord(m_point) 
        #print(self.particles[0].neigh)
    #    list_neigh = np.concatenate((self.particles[0].neigh, self.particles[1].neigh))
        #print(list_neigh,'compare to',[n for n in range(N_tot) if self.q[n] != 0])
   
    #    for n in list_neigh:
    #       coord_n = nToGridCoord(n)
    #        d = BoxScaleDistance( (coord_n - coord_m) * h )
    #        if d != 0:
                #potential = potential + self.q[n] / d
    #            potential = potential + self.q[n] * np.exp(-k_b * d) / (eps * d) 
    #    return potential
    
    # compute the electrostatic potential considering the charges as point like and in their initial position 
    #def ComputeBoundaryPotentialn(self, m_point):
    #    potential = 0
    #    coord_m = nToGridCoord(m_point) 
    #    for particle in self.particles:
    #        d = BoxScaleDistance(particle.pos - coord_m * h)
    #        potential = potential + particle.charge / d 
    #   return potential

    # initialize the field to apply Verlet (specify which step)  
    #def InitializeField(self, step = 1):
    #    if step == 0:
    #        self.phi_prev = np.array([self.ComputePotentialn(m) for m in range(N_tot)])
    #    elif step == 1:
    #        self.phi = np.array([self.ComputePotentialn(m) for m in range(N_tot)])
    #    else:
    #        print("Error: step can only be 0 (initial step) or 1 (1 or following)")
    
    # get the n indices of the internal points of the grid
    #def GetInternaln(self):
    #    list_internal_indices = [m for m in range(N_tot) if IsBoundary(m) == False]
    #    return list_internal_indices
    
    # get the n indices of the boundary points of the grid
    #def GetBoundaryn(self):
    #    list_boundary_indices = [m for m in range(N_tot) if IsBoundary(m) == True]
    #    return list_boundary_indices
    
    # compute the potential for every point of the grid using the spread charges
    #def ComputePhiTh(self):
    #    self.phi_th = np.array([self.ComputePotentialn(m) for m in range(N_tot)])

    #def GetInsideSoluten(self):
    #    n_in = []

    #    for particle in self.particles:
    #        radius = particle.radius
    #        ratio = round(radius / h)
    #        i_list = [particle.pos[0] / h + delta for delta in range(0, ratio)]
    #        j_list = [particle.pos[0] / h + delta for delta in range(0, ratio)]
    #        k_list = [particle.pos[0] / h + delta for delta in range(0, ratio)]
    #        print(i_list)

    #        for i in i_list:
    #            for j in j_list:
    #                for k in k_list:
    #                    if BoxScaleDistance(particle.pos - np.array([i,j,k]) * h) < radius:
    #                        n_in.append(i + j * N + k * N * N)

    #        print(n_in)
    #        return n_in
    
    # exclusion function - 1 if inside the solute, 0 if outside
    #@jit
    '''
    def H(self, n, offset):
        # TO BE FIXED
        if n >= N_tot or n < 0:
            return 0
        
        coord = dict_indices_nToCoord[n] 
        pos = coord * h
        
        for particle in self.particles:
            if particle.radius >= BoxScaleDistance(pos + offset - particle.pos):
                return 1        
    
        return 0


    def UpdateExclusionF(self, n):
        i, j, k = dict_indices_nToCoord[n]
        coord_needed = (np.array([[i, j, k], [i - 1, j, k], [i, j, k], [i, j - 1, k], [i, j, k], [i, j, k - 1], [i, j, k]]) + N ) % N
        
        pos_vector = h * coord_needed + offset_update
        pos_bool = []
        #print(coord_needed)
        for particle in self.particles:
            pos_diff = (pos_vector - particle.pos) - L * np.round((pos_vector - particle.pos) / L)
            pos_norm = np.linalg.norm(pos_diff, axis=-1)
            pos_bool.append(pos_norm <= particle.radius)
        
        pos_bool = np.array(pos_bool)
        if self.N_p == 2:
            result1 = np.logical_or(pos_bool[0], pos_bool[1]).astype(int)
            return result1
        elif self.N_p == 4:
            result1 = np.logical_or(pos_bool[0], pos_bool[1])
            result2 = np.logical_or(pos_bool[2], pos_bool[3])
            result = np.logical_or(result1,result2).astype(int)
            return result
    '''
   
    '''
        radius1 = self.particles[0].radius
        radius2 = self.particles[1].radius
        radius3 = self.particles[2].radius
        radius4 = self.particles[3].radius

        pos_diff1 = (pos_vector - self.particles[0].pos) - L * np.round((pos_vector - self.particles[0].pos) / L)
        pos_diff2 = (pos_vector - self.particles[1].pos) - L * np.round((pos_vector - self.particles[1].pos) / L)
        pos_diff3 = (pos_vector - self.particles[2].pos) - L * np.round((pos_vector - self.particles[2].pos) / L)
        pos_diff4 = (pos_vector - self.particles[3].pos) - L * np.round((pos_vector - self.particles[3].pos) / L)
        
        pos_norm1 = np.linalg.norm(pos_diff1, axis=-1)
        pos_norm2 = np.linalg.norm(pos_diff2, axis=-1)
        pos_norm3 = np.linalg.norm(pos_diff3, axis=-1)
        pos_norm4 = np.linalg.norm(pos_diff4, axis=-1)

        pos_bool1 = pos_norm1 <= radius1
        pos_bool2 = pos_norm2 <= radius2
        pos_bool3 = pos_norm3 <= radius3
        pos_bool4 = pos_norm4 <= radius4
'''

    
'''
# check whether a point is on the boundary of the grid or not
def IsBoundary(n):
    [i, j, k] = nToGridCoord(n)
    
    if i == 0 or i == N - 1 or j == 0 or j == N - 1 or k == 0 or k == N - 1:   
        return True
    else:
        return False
'''

 
###########################################################################################

###### Matrix Initialization functions #######

'''
# periodic boundary conditions
def InitializeCompleteMatrixPBC(grid, M, index):
    n_list_1 = grid.particles[0].ComputeInitialNeighField()
    n_list_2 = grid.particles[1].ComputeInitialNeighField()
    n_list_tot = np.concatenate((n_list_1, n_list_2)).astype(int)#counter = 0
    
    index = np.array(index).astype(int)

    for n in n_list_tot:
        eps_x_n = eps + (1 - eps) * grid.H(n, [h/2, 0, 0])
        eps_x_n_1 = eps + (1 - eps) * grid.H(n - 1, [h/2, 0, 0])

        eps_y_n = eps + (1 - eps) * grid.H(n, [0, h/2, 0])
        eps_y_n_N = eps + (1 - eps) * grid.H(n - N, [0, h/2, 0])

        eps_z_n = eps + (1 - eps) * grid.H(n, [0, 0, h/2])
        eps_z_n_NN = eps + (1 - eps) * grid.H(n - N * N, [0, 0, h/2])

        k_n =  k_bar**2 * (1 - grid.H(n, [0, 0, 0]))

        M[n][n] = - (eps_x_n + eps_x_n_1 + eps_y_n + eps_y_n_N + eps_z_n + eps_z_n_NN) - k_n * h**2
        M[n][index[n][4]] = eps_x_n    #n + 1
        M[n][index[n][2]] = eps_x_n_1  #n - 1
        M[n][index[n][5]] = eps_y_n    #n + N
        M[n][index[n][1]] = eps_y_n_N  #n - N
        M[n][index[n][6]] = eps_z_n    #n + N * N
        M[n][index[n][0]] = eps_z_n_NN #n - N * N
    
    #print(counter, N_tot - counter)
    return M    


def InitializeProvaMatrix_7EntriesPBC(grid, M):   
    for n in range(N_tot):
        result = grid.UpdateExclusionF(n)
        
        eps_x_n = eps + (1 - eps) * result[0]
        eps_x_n_1 = eps + (1 - eps) * result[1]

        eps_y_n = eps + (1 - eps) * result[2]
        eps_y_n_N = eps + (1 - eps) * result[3]

        eps_z_n = eps + (1 - eps) * result[4]
        eps_z_n_NN = eps + (1 - eps) * result[5]

        k_n =  k_bar**2 * (1 - result[6])
        
        M[n][0] = eps_z_n_NN                                                                        # n - N^2
        M[n][1] = eps_y_n_N                                                                         # n - N
        M[n][2] = eps_x_n_1                                                                         # n - 1
        M[n][3] = - (eps_x_n + eps_x_n_1 + eps_y_n + eps_y_n_N + eps_z_n + eps_z_n_NN) - k_n * h**2 # n
        M[n][4] = eps_x_n                                                                           # n + 1
        M[n][5] = eps_y_n                                                                           # n + N
        M[n][6] = eps_z_n                                                                           # n + N^2

    return M


def InitializeMatrix_7EntriesPBC():
    M = eps * np.ones((N_tot, 7))   
    M[:,3] = -6 * M[:,3]

    return M


def UpdateMatrix_7EntriesPBC(grid, M):
    #n_list_1 = grid.particles[0].ComputeNeighField(old_list[0])
    #n_list_2 = grid.particles[1].ComputeNeighField(old_list[1])

    # compute {n} that are forming a cube around the particle so to update only those points
    n_list_tot = []

    for particle in grid.particles:
        n_list_tot.append(particle.ComputeInitialNeighField())
      
    n_list_tot = np.unique(np.array(n_list_tot)).astype(int)
 
    
    for n in n_list_tot:

        eps_x_n = eps + (1 - eps) * grid.H(n, [h/2, 0, 0])
        eps_x_n_1 = eps + (1 - eps) * grid.H(n - 1, [h/2, 0, 0])

        eps_y_n = eps + (1 - eps) * grid.H(n, [0, h/2, 0])
        eps_y_n_N = eps + (1 - eps) * grid.H(n - N, [0, h/2, 0])

        eps_z_n = eps + (1 - eps) * grid.H(n, [0, 0, h/2])
        eps_z_n_NN = eps + (1 - eps) * grid.H(n - N * N, [0, 0, h/2])

        k_n =  k_bar**2 * (1 - grid.H(n, [0, 0, 0]))
        
        M[n][0] = eps_z_n_NN                                                                        # n - N^2
        M[n][1] = eps_y_n_N                                                                         # n - N
        M[n][2] = eps_x_n_1                                                                         # n - 1
        M[n][3] = - (eps_x_n + eps_x_n_1 + eps_y_n + eps_y_n_N + eps_z_n + eps_z_n_NN) - k_n * h**2 # n
        M[n][4] = eps_x_n                                                                           # n + 1
        M[n][5] = eps_y_n                                                                           # n + N
        M[n][6] = eps_z_n                                                                           # n + N^2

    return M
    
    #return M, old_list


def UpdateProvaMatrix_7EntriesPBC(grid, M):
    # compute {n} that are forming a cube around the particle so to update only those points
    n_list_tot = []

    for particle in grid.particles:
        n_list_tot.append(particle.ComputeInitialNeighField())
      
    n_list_tot = np.unique(np.array(n_list_tot)).astype(int)
    
    for n in n_list_tot:
        result = grid.UpdateExclusionF(n)
        eps_x_n = eps + (1 - eps) * result[0]
        eps_x_n_1 = eps + (1 - eps) * result[1]

        eps_y_n = eps + (1 - eps) * result[2]
        eps_y_n_N = eps + (1 - eps) * result[3]

        eps_z_n = eps + (1 - eps) * result[4]
        eps_z_n_NN = eps + (1 - eps) * result[5]

        k_n =  k_bar**2 * (1 - result[6])
        
        M[n][0] = eps_z_n_NN                                                                        # n - N^2
        M[n][1] = eps_y_n_N                                                                         # n - N
        M[n][2] = eps_x_n_1                                                                         # n - 1
        M[n][3] = - (eps_x_n + eps_x_n_1 + eps_y_n + eps_y_n_N + eps_z_n + eps_z_n_NN) - k_n * h**2 # n
        M[n][4] = eps_x_n                                                                           # n + 1
        M[n][5] = eps_y_n                                                                           # n + N
        M[n][6] = eps_z_n                                                                           # n + N^2

    return M  
'''


#index[n][0] k-1
#index[n][1] j-1

def DetIndices_7entries():
    index = np.zeros((N_tot, 7))

    for n in range(N_tot):
        index[n][3] = n

        i, j, k = dict_indices_nToCoord[n]

           
        if i != N - 1:
            index[n][4] = n + 1 #  i + 1
        else:
            index[n][4] = j * N + k * N * N # i + 1 

        if j != N - 1:
            index[n][5] = n + N # j + 1
        else:
            index[n][5] = i + k * N * N
  
        if k != N - 1:
            index[n][6] = n + N * N
        else:
            index[n][6] = i + j * N

        if i != 0: 
            index[n][2] = n - 1
        else:
            index[n][2] = N - 1 + j * N + k * N * N

        if j != 0: 
            index[n][1] = n - N
        else:
            index[n][1] = i + (N - 1) * N + k * N * N

        if k != 0:
            index[n][0] = n - N * N # k - 1
        else:
            index[n][0] = i + j * N + (N - 1) * N * N
      
    return index