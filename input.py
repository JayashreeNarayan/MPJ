import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 5:
    print("\nUsage: python your_script.py <N_p> <N> <N_steps> <L>")
    print("Where:")
    print("<N_p>        : Number of particles (integer)")
    print("<N>          : Number of grid points per side (integer)")
    print("<N_steps>    : Number of steps to perform (integer)")
    print("<L>          : Length of the side of the box (float, Angstrom)\n")
    sys.exit(1)  # Exit the script with an error code

a0 = 0.529177210903 #Angstrom 
t_au = 2.4188843265857 * 1e-2 #fs = 1 a.u. of time 

###################################################################################

### Output settings ###

class OutputSettings:
    print_field = False
    print_performance = False
    print_solute = True
    print_energy = True
    print_temperature = True
    print_tot_force = False
    print_iters = False
    path = 'Outputs/'
    debug = False
    restart = False
    generate_restart_file = False
output_settings = OutputSettings()

###################################################################################

### Grid and box settings ###
class GridSetting:
    N = int(sys.argv[2])              #number of grid points per size
    N_tot = N * N * N        #number of grid points
    L = float(sys.argv[4])  / a0  #20.9/ a0  #6.64 / a0 Np = 8 # 5.6 * 2 / a0   #box lenght
    h = L / N                #mesh size
    path = '../data/paper/diffusion/production/cluster/gamma_1e-3_init_10K/series/run_30/output/'
    input_restart_filename = path + 'restart_N' + str(N) + '_step9999.csv'
    input_filename = 'input_files_new/input_coord' + str(int(sys.argv[1])) + '.csv' 
    N_p = int(sys.argv[1])
grid_setting = GridSetting()

###################################################################################

### MD variables ###
class MDVariables:
    N_steps = int(sys.argv[3])  
    init_steps = 0
    thermostat = False
    dt = 0.25 / t_au        # timestep for the solute evolution given in fs and converted in a.u.
    stride = 1              # saves every stride steps
    tol = 1e-7
    omega = 1 #overrelaxation parameter # REMOVE
    initialization = 'CG'   # always CG
    preconditioning = 'Yes' #Yes or No
    elec = True
    not_elec = True
    potential = 'TF' #TF or LJ
    integrator = 'OVRVO'
    gamma = 1e-3 # OVRVO parameter
    T = 1550 # K
    kB = 3.1668 * 1e-6 #Eh/K
    kBT = kB * T #Eh
md_variables = MDVariables()

