import pandas as pd
import numpy as np

a0 = 0.529177210903 #Angstrom 
N_p = 250
N = 100
#path = '../data/dati_parigi/gr_paper_sara/equilibration/test_thermostat/'
#path = '../data/dati_parigi/gr_paper_sara/equilibration/test_thermostat/'
#path = '../data/test_restart/'
#path = '../data/paper/diffusion/equilibration/run_0813_eq5K/output_10K_15K/'
path = '../data/paper/diffusion/equilibration/N_100/gamma_1e-4/init_5K/'

def generate_restart(iter):

    filename = path + 'solute_N' + str(N) + '.csv'

    df = pd.read_csv(filename)
    m_Na = 22.99
    m_Cl = 35.453
    radius = np.ones(N_p)

    new_df = df[df['iter'] == iter][['charge','x','y','z','vx','vy','vz']] 
    col_mass_bool = new_df['charge'] == 1
    col_mass = [m_Na if bool == True else m_Cl for bool in col_mass_bool]

    new_df.insert(loc=1, column='mass',value=col_mass)
    print(np.shape(new_df))
    new_df.insert(loc=2, column='radius',value=radius)
    new_df['x'] = new_df['x'] * a0
    new_df['y'] = new_df['y'] * a0
    new_df['z'] = new_df['z'] * a0

    print(new_df.head())
    filename_output = path + 'restart_N' + str(N) + '_step' + str(iter) +  '.csv'
    new_df.to_csv(filename_output, index=False)
    return filename_output

generate_restart(4999)