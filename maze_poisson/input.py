from pathlib import Path

import numpy as np
import yaml

from .constants import a0, kB, m_Cl, m_Na, t_au
from .loggers import logger

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
        self._input_file = None
        self._restart_file = None

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

    @property
    def input_file(self):
        if self._input_file is None:
            self._input_file = 'input_files_new/input_coord'+str(self.N_p)+'.csv'
        return self._input_file

    @property
    def restart_file(self):
        if self._restart_file is None:
            self._restart_file = 'restart_files/density_'+str(np.round((self.N_p/2)*(m_Na+m_Cl)*1000 / (self.L_ang*1.e-8)**3, 3))+'/restart_N'+str(self.N)+'_step9999.csv'
        return self._restart_file

###################################################################################

### MD variables ###
class MDVariables:
    def __init__(self):
        self._T = None
        self._kBT = None
        self.N_steps = None
        self.init_steps = None
        self.thermostat = None # to move
        self._dt_fs = None # dt in fs
        self._dt = None        # timestep for the solute evolution given in fs and converted in a.u. # to move
        self.stride = 1              # saves every stride steps
        self.initialization = 'CG'   # always CG
        self.preconditioning = 'Yes' # Yes or No
        self.rescale = None # rescaling of the initial momenta to have tot momenta = 0
        self.elec = None # to move
        self.not_elec = None # to move
        self.potential = 'TF' # Tosi Fumi (TF) or Leonard Jones (LJ)
        self.integrator = 'OVRVO'
        self.gamma = 1e-3 # OVRVO parameter
    
    @property
    def T(self):
        return self._T
    
    @T.setter
    def T(self, value):
        self._T = value
        self._kBT = kB * value
    
    @property
    def kBT(self):
        return self._kBT
    
    @property
    def dt_fs(self):
        return self._dt_fs
    
    @dt_fs.setter
    def dt_fs(self, value):
        self._dt_fs = value
        self._dt = value / t_au
    
    @property
    def dt(self):
        return self._dt

required_inputs = {
    'grid_setting': ['N', 'L', 'N_p'],
    'output_settings': ['restart'],
    'md_variables': ['N_steps', 'tol', 'rescale', 'T']
}

def initialize_from_yaml(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    if not isinstance(filename, Path):
        logger.error("filename must be a Path or a str")
        raise TypeError('filename must be a Path or a str')
        
    if not filename.exists():
        logger.error(f'Input file {filename} does not exist')
        raise FileNotFoundError(f'Input file {filename} does not exist')
    
    grid_setting = GridSetting()
    output_settings = OutputSettings()
    md_variables = MDVariables()

    with filename.open() as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    missing = []
    for key in ['output_settings', 'grid_setting', 'md_variables']:
        ptr = data.get(key, {})
        req = required_inputs.get(key, [])
        missing += [f'{key}.{r}' for r in req if r not in ptr]
        for k, v in data[key].items():
            setattr(eval(key), k, v)

    if missing:
        raise ValueError(f'Missing required inputs: {", ".join(missing)}')

    if output_settings.restart:
        if not grid_setting.restart_file:
            logger.error('restart_file must be provided if restart is True')
            raise ValueError('restart_file must be provided if restart is True')
        if not Path(grid_setting.restart_file).exists():
            logger.error(f'Restart file {grid_setting.restart_file} does not exist')
            raise FileNotFoundError(f'Restart file {grid_setting.restart_file} does not exist')

    return grid_setting, output_settings, md_variables
