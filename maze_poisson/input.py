import sys
from pathlib import Path

import yaml

from .constants import a0, t_au

# Check if the correct number of arguments is provided
# if len(sys.argv) != 5:
#     print("\nUsage: python your_script.py <N_p> <N> <N_steps> <L>")
#     print("Where:")
#     print("<N_p>        : Number of particles (integer)")
#     print("<N>          : Number of grid points per side (integer)")
#     print("<N_steps>    : Number of steps to perform (integer)")
#     print("<L>          : Length of the side of the box (float, Angstrom)\n")
#     sys.exit(1)  # Exit the script with an error code

# a0 = 0.529177210903 #Angstrom 
# t_au = 2.4188843265857 * 1e-2 #fs = 1 a.u. of time 


###################################################################################

### Output settings ###

class OutputSettings:
    print_field = True
    print_performance = False
    print_solute = True
    print_energy = True
    print_temperature = True
    print_tot_force = True
    print_iters = False
    path = 'Outputs/'
    debug = False
    restart = True
    generate_restart_file = False
# output_settings = OutputSettings()

###################################################################################

### Grid and box settings ###
class GridSetting:
    # N = int(sys.argv[2])              #number of grid points per size
    # N_tot = N * N * N        #number of grid points
    # L = float(sys.argv[4])  / a0  #20.9/ a0  #6.64 / a0 Np = 8 # 5.6 * 2 / a0   #box lenght
    # h = L / N                #mesh size
    # path = './restart_files/'
    # input_restart_filename = path + 'restart_N' + str(N) + '_step9999.csv'
    # input_filename = 'input_files_new/input_coord' + str(int(sys.argv[1])) + '.csv'
    # N_p = int(sys.argv[1])

    def __init__(self):
        self._N = None
        self._L = None
        # self.h = None
        self.N_p = None
        self._N_tot = None
        self._h = None

        self.input_file = None
        self.restart_file = None

    @property
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        self._N = value
        self._N_tot = int(value ** 3)
        self._h = None
        # self.input_restart_filename = self.path + 'restart_N' + str(self.N) + '_step9999.csv'
        # self.input_filename = 'input_files_new/input_coord' + str(self.N_p) + '.csv'

    @property
    def N_tot(self):
        return self._N_tot
    
    # @N_tot.setter
    # def N_tot(self, value):
    #     fl = value ** (1/3)
    #     if fl.is_integer():
    #         self.N = int(fl)
    #     else:
    #         raise ValueError('N_tot must be a perfect cube')
        
    @property
    def L(self):
        return self._L
    
    @L.setter
    def L(self, value):
        self._h = None
        self._L = value / a0

    @property
    def h(self):
        if self._h is None:
            self._h = self.L / self.N
        return self._h

# grid_setting = GridSetting()

###################################################################################

### MD variables ###
class MDVariables:
    # N_steps = int(sys.argv[3])  
    N_steps = None
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
# md_variables = MDVariables()

required_inputs: dict[str, list[str]] = {
    'grid_setting': ['N', 'L', 'N_p', 'input_file'],
    'output_settings': [],
    'md_variables': ['N_steps']
}

def initialize_from_yaml(filename: Path | str):
    if isinstance(filename, str):
        filename = Path(filename)
    if not isinstance(filename, Path):
        raise TypeError('filename must be a Path or a str')
    if not filename.exists():
        raise FileNotFoundError(f'Input file {filename} does not exist')
    
    grid_setting = GridSetting()
    output_settings = OutputSettings()
    md_variables = MDVariables()

    with filename.open() as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    for key in ['output_settings', 'grid_setting', 'md_variables']:
        if key in data:
            for k, v in data[key].items():
                setattr(eval(key), k, v)

    if output_settings.restart:
        if not grid_setting.restart_file:
            raise ValueError('restart_file must be provided if restart is True')
        if not Path(grid_setting.restart_file).exists():
            raise FileNotFoundError(f'Restart file {grid_setting.restart_file} does not exist')
        
    missing = []
    for key, value in required_inputs.items():
        for v in value:
            if not getattr(eval(key), v):
                missing.append(f'{key}.{v}')

    if missing:
        raise ValueError(f'Missing required inputs: {", ".join(missing)}')

    return grid_setting, output_settings, md_variables

