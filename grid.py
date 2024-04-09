from input import grid_setting, a0, md_variables, output_settings
from particle import Particle, g
import numpy as np
from indices import dict_indices_nToCoord
import pandas as pd
from linkedcell import LinkedCell

if output_settings.print_energy:
    from output_md import file_output_energy

if output_settings.print_temperature:
    from output_md import file_output_temperature


# get input variables from input file
N = grid_setting.N
N_tot = grid_setting.N_tot
N_p = grid_setting.N_p
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
T = md_variables.T
kB = md_variables.kB

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
        self.temperature = T
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


    def Energy(self, iter, print_energy, prev=False):
        
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


        self.energy = kinetic + pot_tot
        if print_energy:
            file_output_energy.write(str(iter * dt) + ',' +  str(self.energy) + ',' +  str(kinetic) + ',' + str(pot_tot) + '\n')#+ potential) + '\n')


    def Temperature(self, iter, print_temperature):
        mi_vi2 = [p.mass * np.dot(p.vel, p.vel) for p in self.particles]
        self.temperature = np.sum(mi_vi2) / (3 * N_p * kB)
        
        if print_temperature:
            file_output_temperature.write(str(iter * dt) + ',' +  str(self.temperature) + '\n')




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