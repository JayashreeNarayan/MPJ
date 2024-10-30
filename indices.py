from input import grid_setting
import numpy as np

N = grid_setting.N
N_tot = grid_setting.N_tot
a0 = 0.529177210903 #Angstrom 


# translate index n = i + j * N + k * N * N  to a set of indices (i,j,k)
def nToGridCoord(n, N): # COULD BE INVERTED FROM I,J,K TO n - COULD BE OPT
    i = n % N
    j = ((n - i) / N) % N
    k = ((n - i) / N - j) / N

    coord = np.array([i, j, k]).astype(int)
    if n == i + j * N + k * N * N:
        return coord
    else:
        print('Conversion gone wrong: check nToGridCoord(n) function')
        print(n, '!=', i + j * N + k * N * N)
        print(coord)

dict_indices_nToCoord = { n : nToGridCoord(n, N) for n in range(N_tot)} 
dict_indices_CoordTon = { tuple(nToGridCoord(n, N)): n  for n in range(N_tot)} 

def GetDictTF(): # get dictionary of TF params
    charge_totNaNa = 2
    charge_totNaCl = 0
    charge_totClCl = -2

    A_NaCl = 7.7527 * 1e-3 #Hartree = 20.3548 kJ/mol 
    C_NaCl = 0.2569 / a0**6 # = 674.4798 kj/mol * Ang^6 
    D_NaCl = 0.3188 / a0**8 # = 837.0777 kj/mol * Ang^8 
    sigma_TF_NaCl = 2.755 / a0

    A_NaNa = 9.6909 * 1e-3 #Hartree = 25.4435 kJ/mol 
    C_NaNa = 0.0385 / a0**6 # = 101.1719 kj/mol * Ang^6 
    D_NaNa = 0.0183 / a0**8 # = 48.1771 kj/mol * Ang^8 
    sigma_TF_NaNa = 2.340 / a0 # 2.340 angs
    
    A_ClCl = 5.8145 * 1e-3 #Hartree = 15.2661 kJ/mol 
    C_ClCl = 2.6607 / a0**6 # = 6985.6841 kj/mol * Ang^6 
    D_ClCl = 5.3443 / a0**8 # = 14,031.5897 kj/mol * Ang^8 
    sigma_TF_ClCl = 3.170 / a0 # 3.170 angs
    
    dictTF = {charge_totNaCl: [A_NaCl, C_NaCl, D_NaCl, sigma_TF_NaCl], 
              charge_totNaNa: [A_NaNa, C_NaNa, D_NaNa, sigma_TF_NaNa], 
              charge_totClCl: [A_ClCl, C_ClCl, D_ClCl, sigma_TF_ClCl]
              } 
    return dictTF

