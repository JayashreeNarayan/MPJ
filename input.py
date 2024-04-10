import numpy as np
import sys


N_input = int(sys.argv[2]) 
N_steps_input = int(sys.argv[3]) 
a0 = 0.529177210903 #Angstrom 

###################################################################################

### Grid and box settings ###
class GridSetting:
    N = N_input              #number of grid points per size
    N_tot = N * N * N        #number of grid points
    L = 13.27 / a0 #13.27 / a0 ###6.64 / a0 Np = 8 # 5.6 * 2 / a0   #box lenght
    h = L / N                #mesh size
    input_filename = 'input_files/input_coord64.csv'
    N_p = int(sys.argv[1]) 
grid_setting = GridSetting()

###################################################################################

### MD variables ###
class MDVariables:
    N_steps = N_steps_input   
    dt = 10              # timestep for the solute evolution 
    #m = 1.0                 # mass in atomic units
    stride = 1              # saves every stride steps
    dx = 1e-3
    delta = np.array([-dx, 0., 0.]) # step size for particle 0 in the x, y and z directions [-0.01]
    tol = 1e-7
    omega = 1  #overrelaxation parameter
    initialization = 'CG'   # Can be 'CG' or 'none' - to do first two steps for Verlet
    preconditioning = 'Yes' #Yes or No
    elec = True
    not_elec = True
    potential = 'TF' #TF or LJ
    T = 1539 # K
    kB = 3.1668 * 1e-6 #Eh/K
    kBT = kB * T #Eh
md_variables = MDVariables()

###################################################################################

### Output settings ###

class OutputSettings:
    print_field = False
    print_performance = False
    print_solute = False
    print_energy = True
    print_temperature = True
    path = '../data/test_github/'

output_settings = OutputSettings()
