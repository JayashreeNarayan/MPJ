import numpy as np
import pandas as pd
from.constants import a0

def generate_restart(grid_setting, output_settings):
    N_p = grid_setting.N_p
    N = grid_setting.N
    path = output_settings.path
    filename = path + 'solute_N' + str(N) + '.csv'

    df = pd.read_csv(filename)
    m_Na = 22.99
    m_Cl = 35.453
    radius = np.ones(N_p)

    max = np.max(df['iter'])
    new_df = df[df['iter'] == max][['charge','x','y','z','vx','vy','vz']] 
    col_mass_bool = new_df['charge'] == 1
    col_mass = [m_Na if bool == True else m_Cl for bool in col_mass_bool]

    new_df.insert(loc=1, column='mass',value=col_mass)
    print(np.shape(new_df))
    new_df.insert(loc=2, column='radius',value=radius)
    new_df['x'] = new_df['x'] * a0
    new_df['y'] = new_df['y'] * a0
    new_df['z'] = new_df['z'] * a0

    print(new_df.head())
    filename_output = path + 'restart_N' + str(N) + '_5K.csv'
    new_df.to_csv(filename_output, index=False)
    return filename_output
