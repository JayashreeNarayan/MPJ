import numpy as np
import pandas as pd
from indices import dict_indices_nToCoord, nToGridCoord
from scipy.interpolate import CubicSpline
from input import grid_setting
import math
import matplotlib.pyplot as plt

N = grid_setting.N
h = grid_setting.h
N_tot = grid_setting.N_tot
L = grid_setting.L

def spline_analysis(grid, output_file):
    #V = 27.211386245988
    df = pd.read_csv(output_file)
    phi_nospline = df['phi'] # N = 100
    #x_nospline = df['x'] #x for N = 100
    
    N_nospline = 100
    N_tot_nospline = N_nospline * N_nospline * N_nospline
    h_nospline = L / N_nospline
    
    dict_indices_nToCoord_nospline = { n : nToGridCoord(n, N_nospline) for n in range(N_tot_nospline)} 
    dict_indices_CoordTon_nospline = { tuple(nToGridCoord(n, N_nospline)): n  for n in range(N_tot_nospline)} 

    particle_neigh = []

    for p in grid.particles:
        indices = [math.floor(p.pos[i] / h_nospline)  for i in range(3)]
        neigh_indices = []        

        for i in range(indices[0], indices[0] + 2):
            for j in range(indices[1], indices[1] + 2):
                for k in range(indices[2], indices[2] + 2):
                    n_pos = dict_indices_CoordTon_nospline[tuple([i % N_nospline, j % N_nospline, k % N_nospline])] 
                    neigh_indices.append(n_pos)
        
        particle_neigh.append(np.array(neigh_indices).astype(int))

    err = 0

    for l, particle in enumerate(grid.particles):
        for l_n, n in enumerate(particle.neigh):
            i, j, k = dict_indices_nToCoord[n]
            i_vec = np.array([i - 2, i - 1, i, i + 1, i + 2]) 
            #i_vec = np.array([i - 1, i, i + 1])
            
            x_vec = i_vec * h
            phi_x = np.array([grid.phi[p % N + j * N + k * N * N] for p in i_vec])
            
            # cubic spline over given points
            cs_x = CubicSpline(x_vec, phi_x)

            i_ns, j_ns, k_ns = dict_indices_nToCoord_nospline[particle_neigh[l][l_n]]
            i_test = np.array([i_ns - 4, i_ns - 3, i_ns - 2, i_ns - 1, i_ns, i_ns + 1, i_ns + 2, i_ns + 3, i_ns + 4])
            x_test = i_test * h_nospline
           
            phi_spline = np.array(cs_x(x_test)) #campo interpolato sui punti che sono 
            list_neigh_ns = [dict_indices_CoordTon_nospline[tuple([p % N_nospline, j_ns % N_nospline, k_ns % N_nospline])] for p in i_test]
            phi_x_nospline = np.array([df[df['n'] == n]['phi'] for n in list_neigh_ns])
            
            err = err + np.sum([np.abs(phi_spline[m] - phi_x_nospline[m]) for m in range(len(phi_spline))])

            #for m in range(len(phi_spline)):
                #print(phi_spline[m], phi_x_nospline[m], np.abs(phi_spline[m] - phi_x_nospline[m]))
            
            plt.plot(x_test, phi_spline, label='spline')
            plt.plot(x_test, phi_x_nospline, label='nospline')
            plt.legend()
            plt.show()
    

    #print(phi_spline[0])
    N_pts = grid.N_p * 8 * len(phi_spline)
    #print(phi_x_nospline[0] - phi_spline[0])
   

    err = err / N_pts
    print(err)
    return 1

