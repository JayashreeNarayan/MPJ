import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ...constants import a0
from scipy.optimize import curve_fit

N_vector = [30,40,50,60,70,80,90,100] # will change from 10 to 100 once the CG works for 10,20
N_p_vector = [2,16,54,128,216,432,686,1024]
filename_MaZe = 'Outputs/performance_N'

def g(x,a,b):
    return a * x**b

def g1(x, a, b):
    return a * x + b

def f(x,a,b):
    return a * x**b * np.log(x**b)

# plot of time/iter VS N_grid = N^3
def plot_time_iterNgrid(filename_MaZe=filename_MaZe,  title='time_per_iter_VS_N_grid', N_vector=N_vector, data1 = "time", data2='n_iters'):
    N_vector = np.array(N_vector)
    df_list_MaZe = [pd.read_csv(filename_MaZe + str(i) + '.csv') for i in N_vector]

    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][100:]/df[data2][100:]))
        sd1.append(np.std(df[data1][100:]/df[data2][100:]))
        
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_vector
    poptMaZe, _ = curve_fit(g1, x**3, avg1, sigma = sd1, absolute_sigma=True, p0=[0.1, 1.4])
    a_optMaZe, b_optMaZe = poptMaZe

    print(f'Optimized parameters MaZe t: a = {a_optMaZe}, b = {b_optMaZe}')

    plt.figure(figsize=(10, 8))
    plt.errorbar(x**3, avg1,sd1, label = 'MaZe', color='r',marker='o', linestyle='', linewidth=1.5, markersize=6,capsize=4)
    plt.plot(x**3, g1(x**3, a_optMaZe, b_optMaZe), label=f'fit $ax+b$, b = {b_optMaZe:.4f},  a = {a_optMaZe:.4f}')
    plt.xlabel('Number of grid points', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    name =  title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()


def f(x,a,b):
    return a * x + b

# plot of n iterations VS N_grid = N^3
def plot_convNgrid(filename_MaZe=filename_MaZe, N_vector=N_vector, title='n_iterations_vs_N_grid' ,data1 = "n_iters"):
    N_vector = np.array(N_vector)

    df_list_MaZe = [pd.read_csv(filename_MaZe + str(i) + '.csv') for i in N_vector]

    avg1 = []
    sd1 = []

    avg2 = []
    sd2 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
        #print(avg1,sd1)
        avg2.append(np.mean(df[data1][3:]))
        sd2.append(np.std(df[data1][3:]))
        
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)
    avg2 = np.array(avg2)
    sd2 = np.array(sd2)

    x = N_vector**3
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(N_vector**3, avg1,sd1, label = 'MaZe', color='r',marker='o', linestyle='', markersize=6,capsize=4)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe),  label=f'fit $ax^b$, b = {b_optMaZe:.2f} $\\approx 1/3$')
    plt.xlabel('Number of grid points', fontsize=18)
    plt.ylabel('Number of iterations', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    name = title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()


#### PLOT FUNCTION OF Number of particles ####

# plot time / n iterations VS number of particles
def plot_scaling_particles_time_iters(filename_MaZe=filename_MaZe, N_p=N_p_vector, title='time_per_n_iterations_vs_N_p',  N=N_vector, data1 = "time", data2 = "n_iters"):
    N_p = np.array(N_p)
    df_list_MaZe = [pd.read_csv(filename_MaZe + str(N[i]) + '.csv') for i, n in enumerate(N_p)]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]/df[data2][-50:]))
        sd1.append(np.std(df[data1][-50:]/df[data2][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_p
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True, p0=[1e-8,1])
    a_optMaZe, b_optMaZe = poptMaZe

    print(f'Optimized parameters MaZe: a = {a_optMaZe}, b = {b_optMaZe}')
    plt.figure(figsize=(10, 8))
   
    plt.errorbar(N_p, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe),  label=f'fit $ax^b$, b = {b_optMaZe:.2f}')
    plt.xlabel('Number of particles', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    name =  title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()


def f(x,a,b):
    return a * np.log(x)**b


# plot n iterations VS number of particles
def plot_scaling_particles_conv(filename_MaZe, N_p, title,  N=N_vector, data1 = "n_iters"):
    N_p = np.array(N_p)
    df_list_MaZe = [pd.read_csv(filename_MaZe + '/performance_N' + str(N[i]) + '.csv') for i, n in enumerate(N_p)]
    avg1 = []
    sd1 = []


    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]))
        sd1.append(np.std(df[data1][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_p
    poptMaZe, _ = curve_fit(f, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    print(N_p_vector)
    print(sd1)
    plt.errorbar(N_p, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, f(x, a_optMaZe, b_optMaZe), label=f'fit $a\\log{{x}}^b$, b = {b_optMaZe:.2f}') 
    plt.xlabel('Number of particles', fontsize=18)
    plt.ylabel('# Iterations', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    name =  title + ".pdf"
    plt.savefig(name, format='pdf')
    plt.show()
