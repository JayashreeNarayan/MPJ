from pathlib import Path
import yaml
from .constants import a0, kB, t_au

###################################################################################

### Output settings ###

class OutputSettings:
    print_field = None # to move
    print_performance = None # to move
    print_solute = None # to move
    print_energy = None # to move
    print_temperature = None # to move
    print_tot_force = None # to move
    print_iters = False
    path = 'Outputs/'
    debug = False
    restart = None
    generate_restart_file = None # to move
    iter_restart = None

###################################################################################

### Grid and box settings ###
class GridSetting:
    def __init__(self):
        self._N = None
        self._L = None
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

    @property
    def N_tot(self):
        return self._N_tot
        
    @property
    def L(self):
        return self._L
    
    @L.setter
    def L(self, value):
        self._h = None
        self.L_ang = value
        self._L = value / a0

    @property
    def h(self):
        if self._h is None:
            self._h = self.L / self.N
        return self._h

###################################################################################

### MD variables ###
class MDVariables:
    N_steps = None
    init_steps = None
    thermostat = None # to move
    dt_fs = 0.25 # dt in fs
    dt = dt_fs / t_au        # timestep for the solute evolution given in fs and converted in a.u. # to move
    stride = 1              # saves every stride steps
    initialization = 'CG'   # always CG
    preconditioning = 'Yes' # Yes or No
    rescale = None # rescaling of the initial momenta to have tot momenta = 0
    elec = None # to move
    not_elec = None # to move
    potential = 'TF' # Tosi Fumi (TF) or Leonard Jones (LJ)
    integrator = 'OVRVO'
    gamma = 1e-3 # OVRVO parameter
    T = 1550 # K # to move
    kBT = kB * T # E_h

required_inputs = {
    'grid_setting': ['N', 'L', 'N_p', 'input_file'],
    'output_settings': ['restart'],
    'md_variables': ['N_steps', 'tol', 'rescale', 'T']
}

def initialize_from_yaml(filename):
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

