import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...constants import a0, t_au
from . import get_N

#from input import md_variables, grid_setting, a0, output_settings


# N = 100
# folder='Outputs/'
# path='Outputs/'
# file_T = 'temperature_N'
# t_au = 2.4188843265857 * 1e-2 #fs = 1 a.u. of time 
# dt = 0.25 / t_au

def PlotT(filename, dt, label='iter'):
    df = pd.read_csv(filename)
    N = get_N(filename)

    dt /=  t_au

    T = df['T'][1:]
    iter = df[label][1:]
    rel_err = np.std(T)/np.mean(T)
    print(np.shape(T))
    print('\nmean:', np.mean(T))
    print('std:', np.std(T))
    print('relative error:', rel_err)
    plt.figure(figsize=(15, 6))
    plt.plot(iter, T, marker='.', color='red', markersize=5, label='T - mean value = ' + str(np.mean(T)) + '$\pm$' + str(np.std(T)))
    plt.title('Temperature - dt =' + str(dt) + ' fs - N =' + str(N), fontsize=22)
    plt.xlabel('iter', fontsize=15)
    plt.ylabel('T (K)', fontsize=15)
    plt.axhline(1550)
    plt.legend()
    plt.savefig('T_N' + str(N) + '_dt' + str(dt) + '.pdf', format='pdf')
    plt.show()

def plot_Etot_trp(path, N, dt, N_th, L=20.9 / a0, upper_lim=None):
    # File paths
    work_file = path + 'work_trp_N' + str(N) + '.csv'
    df_E = pd.read_csv(path + 'energy_N' + str(N) + '.csv')

    # Energy file columns
    K = df_E['K']
    V_notelec = df_E['V_notelec']
    N_steps_energy = len(df_E)
    recompute_work = False

    # Check if the work file exists and has the correct number of lines
    if os.path.exists(work_file):
        work_df = pd.read_csv(work_file)
        if len(work_df) == N_steps_energy:

            # If the file exists and has the correct number of rows, use the work data
            Ework = work_df['work'].tolist()
            print(f"Work data loaded from {work_file}")
        else:
            print(f"Work file exists but has incorrect number of lines. Recomputing work.")
            recompute_work = True
    else:
        recompute_work = True
    df = pd.read_csv(path + 'solute_N' + str(N) + '.csv')
    iterations = df['iter']
    N_steps = int(iterations.max() + 1)
    print('N_steps =', N_steps)
    if recompute_work:

        # If the file doesn't exist, compute the work
        # df = pd.read_csv(path + 'solute_N' + str(N) + '.csv')
        Np = int(df['particle'].max() + 1)
        df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]
        # iterations = df['iter']
        # N_steps = int(iterations.max() + 1)
        Ework = np.zeros(N_steps)
        work = np.zeros((Np, N_steps))
        # print('N_steps =', N_steps)
        print('Np =', Np)

        # Precompute steps and avoid repeated indexing
        for p in range(Np):
            df_p = df_list[p]
            
            # Calculate displacement between consecutive steps
            delta = np.diff(df_p[['x', 'y', 'z']].values, axis=0)

            # Accumulate the corrected displacements
            c = np.cumsum(delta, axis=0)

            # Compute work for each particle p
            work[p][1:] = - np.trapz(df_p[['fx', 'fy', 'fz']].values, x=c, axis=0)
                
        # Sum up the work across all particles
        Ework = np.add.reduce(work, axis=0)

        # Create the work file with header "iter,work" and save the computed work
        work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
        work_df.to_csv(work_file, index=False)
        print(f"Work data saved to {work_file}")

    # Compute total energy
    E_tot = K + V_notelec + Ework
    mean = np.mean(E_tot[:upper_lim])
    std = np.std(E_tot[:upper_lim])
    print(mean, std)
    mean_K = np.mean(K)
    mean_V_notelec = np.mean(V_notelec)
    mean_work = np.mean(Ework)
    iterations = range(N_steps)
    iter_E = df_E['iter']

    # Plotting the distance and kinetic energy in subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))

    # Plotting kinetic energy
    ax1.plot(iter_E[N_th:upper_lim], K[N_th:upper_lim], marker='.', color='b', markersize=5, label=f'Kinetic energy - $|\\frac{{<K>}}{{<V_{{elec}}>}}| ={np.abs(mean_K/mean_work):.4f}$')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('Contributions to the total energy of the system - N = ' + str(N) + ', dt = ' + str(dt) + ' fs')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plotting potential contributions
    ax2.plot(iterations[N_th:upper_lim], Ework[N_th:upper_lim], marker='.', linestyle='-', color='red', markersize=1, label='Elec')
    ax2.plot(iterations[N_th:upper_lim], V_notelec[N_th:upper_lim], marker='.', linestyle='-', color='mediumturquoise', markersize=1, label=f'Not elec - $|\\frac{{<V_{{not elec}}>}}{{<V_{{elec}}>}}| ={np.abs(mean_V_notelec/mean_work):.4f}$')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Potential')
    ax2.set_title('Potential contributions')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plotting total energy
    ax3.plot(iterations[N_th:upper_lim], E_tot[N_th:upper_lim], marker='.', color='lightgreen', markersize=5, label=f'Tot energy - rel err = {std/mean:.5f}, $\Delta E / \Delta V =$ {std/ np.std(Ework)}')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Tot E')
    ax3.set_title('Total energy over iterations')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(path + '/Energy_analysis_trp_N' + str(N) + '_dt' + str(dt) + '.pdf', format='pdf')
    plt.show()

def plot_work_trp(path, N, N_th, L=20.9 / a0):
    # Check if the work file already exists
    work_file = path + 'work_N' + str(N) + '.csv'
    if os.path.exists(work_file):
        # If the file exists, read the work from it
        work_df = pd.read_csv(work_file)
        Ework = work_df['work'].tolist()
        N_steps = len(Ework)
        print(f"Work data loaded from {work_file}")
    else:
        # If the file doesn't exist, compute the work
        df = pd.read_csv(path + 'solute_N' + str(N) + '.csv')
        Np = int(df['particle'].max() + 1)
        df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]
        iterations = df['iter']
        N_steps = int(iterations.max() + 1)
        Ework = np.zeros(N_steps)
        work = np.zeros((Np, N_steps))
        print('N_steps =', N_steps)
        print('Np =', Np)
        # Precompute steps and avoid repeated indexing
        for p in range(Np):
            df_p = df_list[p]
            x = df_p['x'].values
            y = df_p['y'].values
            z = df_p['z'].values
            fx = df_p['fx'].values
            fy = df_p['fy'].values
            fz = df_p['fz'].values
            # Initialize cumulative displacement
            cx = np.zeros(N_steps)
            cy = np.zeros(N_steps)
            cz = np.zeros(N_steps)
            for i in range(1, N_steps):
                # Calculate displacement between consecutive steps
                dx = x[i] - x[i-1]
                dy = y[i] - y[i-1]
                dz = z[i] - z[i-1]
                # Apply minimum image convention to account for PBC
                dx -= np.rint(dx / L) * L
                dy -= np.rint(dy / L) * L
                dz -= np.rint(dz / L) * L
                # Accumulate the corrected displacements
                cx[i] = cx[i-1] + dx
                cy[i] = cy[i-1] + dy
                cz[i] = cz[i-1] + dz
                # Compute work for each particle p at each step i
                work[p][i] = - (np.trapz(fx[:i], x=cx[:i]) +
                                np.trapz(fy[:i], x=cy[:i]) +
                                np.trapz(fz[:i], x=cz[:i]))
        # Sum up the work across all particles
        Ework = np.add.reduce(work, axis=0)
        # Create the work file with header "iter,work" and save the computed work
        work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
        work_df.to_csv(work_file, index=False)
        print(f"Work data saved to {work_file}")
    # Plotting the work
    iterations = range(N_steps - 1)
    plt.figure(figsize=(15, 6))
    plt.plot(iterations[N_th:], Ework[N_th:-1], marker='.', color='red', markersize=5, label='Potential energy')
    plt.title('Total Work', fontsize=22)
    plt.xlabel('iter', fontsize=15)
    plt.ylabel('Work ($E_H$)', fontsize=15)
    plt.legend(loc='upper left')
    plt.savefig(path + '/Work_trp_N' + str(N) + '.pdf', format='pdf')
    plt.show()


# PlotT(file_T, folder, dt, N)
# plot_Etot_trp(path, N, dt, N_th=0, L=20.9 / a0)
# plot_work_trp(path, N, N_th=0)






