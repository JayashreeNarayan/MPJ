import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

N =100

def plot_force(folder, N, dt):
    df = pd.read_csv(folder + '/tot_force_N' + str(N) + '.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(df['iter'] * dt / 1000, df['Fx'], 'r', linestyle='-', label='Fx')
    plt.plot(df['iter'] * dt / 1000, df['Fy'], 'b', linestyle='-', label='Fy')
    plt.plot(df['iter'] * dt / 1000, df['Fz'], 'g', linestyle='-', label='Fz')
    plt.ylabel('Total force [a.u.]')
    plt.xlabel('Time [ps]')
    plt.title('Total force acting on the system for dt = ' + str(dt) + ' and N = ' + str(N))
    #plt.axhline(np.mean(df['Fx']), label='avg = '+str(np.mean(df['Fx'])))
    #plt.axhline(np.mean(df['Fy']), label='avg = '+str(np.mean(df['Fy'])))
    #plt.axhline(np.mean(df['Fz']), label='avg = '+str(np.mean(df['Fz'])))
    
    title = '/force_tot_N_' + str(N) + '_dt_' + str(dt)
    name = folder + title + ".pdf"
    plt.legend()
    plt.savefig(name, format='pdf')
    plt.show()

def plot_forcemod(folder, N, dt):
    df = pd.read_csv(folder + '/tot_force_N' + str(N) + '.csv')
    Fx = df['Fx']
    Fy = df['Fy']
    Fz = df['Fz']
    Fmod = np.sqrt(Fx**2 + Fy**2 + Fz**2)
    plt.plot(df['t'], Fmod, '.r', linestyle='-', label='N = ' + str(N))
    plt.ylabel('Total force on x axis (a.u.)')
    plt.xlabel('t (a.u.)')
    plt.axhline(np.mean(Fmod), label='avg = '+str(np.mean(Fmod)))
    plt.title('Total force acting on the system for dt = ' + str(dt) + ' and N = ' + str(N))
    title = '/force_totmod_N_' + str(N) + '_dt_' + str(dt)
    name = folder + title + ".pdf"
    plt.legend()
    plt.savefig(name, format='pdf')
    plt.show()

def plot_force_allN(folder, N_list, dt):
    df_list = [pd.read_csv(folder + '/tot_force_N' + str(N) + '.csv') for N in N_list]
    color_list = ['r', 'm', 'b']
    linestyle_list = ['-', '--']
    plt.figure(figsize=(15, 6))

    for i, df in enumerate(df_list):
        plt.plot(df['iter'] * dt / 1000, df['Fx'], '.' + color_list[0], label = 'Fx - N = ' + str(N_list[i]), linestyle=linestyle_list[i])
        plt.plot(df['iter'] * dt / 1000, df['Fy'], '.' + color_list[1], label = 'Fy - N = ' + str(N_list[i]), linestyle=linestyle_list[i])
        plt.plot(df['iter'] * dt / 1000, df['Fz'], '.' + color_list[2], label = 'Fz - N = ' + str(N_list[i]), linestyle=linestyle_list[i])
        
    plt.ylabel('Total force [a.u.]')
    plt.xlabel('Time [ps]')
    plt.title('Total force acting on the system for dt = ' + str(dt))
    plt.legend()
    title = '/force_tot_all_dt_' + str(dt)
    name = folder + title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()


def plot_force_alldt(folder, dt_list, N):
    df_list = [pd.read_csv(folder + '/dt_' + str(dt) + '/tot_force_N' + str(N) + '.csv') for dt in dt_list]
    color_list = ['r', 'm', 'b']
    
    for i, df in enumerate(df_list):
        plt.plot(df['t'] / dt_list[i], df['Fx'], '.' + color_list[i], label = 'dt = ' + str(dt_list[i]), linestyle='-')
    
    plt.ylabel('Total force on x axis (a.u.)')
    plt.xlabel('Iter')
    plt.title('Total force acting on the system for N = ' + str(N))
    plt.legend()
    title = '/force_tot_all_N_' + str(N)
    name = folder + title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()

#folder_force = 'test_Poisson'
#folder_force = 'dati_parigi/gr_paper_sara/equilibration/test_thermostat/fix_th/'
#folder_force = 'test_newE/test_Np2/dt_0.025/'
#plot_force(folder_force, 20, 0.025)
#plot_force(folder_force, 50, 10)
#plot_force(folder_force, 100, 200)
#plot_forcemod(folder_force, 20, 10)
#plot_forcemod(folder_force, 50, 10)
#plot_forcemod(folder_force, 100, 10)
#plot_force_allN(folder_force, [20,50,100], 10)

#plot_force_allN(folder_force, [0.1, 1, 10], 20)
#plot_force_allN(folder_force, [0.1, 1, 10], 50)
#plot_force_allN(folder_force, [0.1, 1, 10], 100)

def plot_force_allomega(folder, omega_list):  
    df_list = [pd.read_csv(folder + 'omega_' + str(omega) + '/tot_force_N50.csv') for omega in omega_list]
    color_list = ['r', 'm', 'b']
    for i, df in enumerate(df_list):
        Fx = df['Fx']
        Fy = df['Fy']
        Fz = df['Fz']
        Fmod = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        plt.plot(df['t'], Fmod, '.' + color_list[i], label = 'omega = ' + str(omega_list[i]) + ', mean = ' + str(np.mean(Fmod)), linestyle='-')
    
    plt.ylabel('Mod of total force (a.u.)')
    plt.xlabel('t (a.u.)')
    plt.title('Total force acting on the system')
    plt.legend()
    title = '/force_tot_all_omega'
    name = folder + title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()

#path = 'test_omega/'
#plot_force_allomega(path, ['1', '1e-2', '1e-4'])


path = 'test_momentum/species_scaling/dt_0.25/'
#plot_force(path, 50, 0.25)
#plot_force(path, 100, 0.25)
#plot_force_allN(path, [50, 100], 0.25)

path = 'test_momentum/species_scaling/dt_0.1/'
#plot_force(path, 50, 0.1)
#plot_force(path, 100, 0.1)
#plot_force_allN(path, [50, 100], 0.1)

path = 'Outputs/'
plot_force(path, N, 0.25)