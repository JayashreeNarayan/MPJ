from input import grid_setting, md_variables
import numpy as np
import time 
#from scipy.sparse import linalg


h = grid_setting.h
L = grid_setting.L
N_tot = grid_setting.N_tot
N = grid_setting.N
tol = md_variables.tol
dt = md_variables.dt
omega = md_variables.omega
preconditioning = md_variables.preconditioning
elec = md_variables.elec
not_elec = md_variables.not_elec
N_p = grid_setting.N_p
T = md_variables.T
kB = md_variables.kB

def VerletSolutePart1(particles, dt=dt):
    for particle in particles:
        #particle.vel = particle.vel * lambda_scaling
        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
        particle.pos = particle.pos + particle.vel * dt 
        particle.pos = particle.pos - L * np.floor(particle.pos / L)   
    return particles


def VerletSolutePart2(grid, old_pos, dt=dt, prev=False):
    grid.potential_notelec = 0
    if not_elec:
        #grid.ComputeForceNotEleLC() #TF or LJ
        grid.ComputeForceNotElecBasic()
    
    tot_force = 0
    for i,particle in enumerate(grid.particles):
        if elec:
            #particle.ComputeForce(grid, prev=prev)
            particle.ComputeForce_FD(grid, prev=prev)
            

        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
        #tot_force = tot_force + particle.force + particle.force_notelec
    
    #print('tot force = ', tot_force)
    return grid



def MatrixVectorProduct_7entries_1(v, index):
    M = np.array([1,1,1,-6,1,1,1])    
    v_matrix = v[index]
    result = np.dot(v_matrix,M)

    return result

#apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoisson(grid,y):
    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = np.copy(tmp)

    # compute the constraint with the provisional value of the field phi
    matrixmult = MatrixVectorProduct_7entries_1(grid.phi, grid.indices7)
    sigma_p = omega * (grid.q / h + matrixmult / (4 * np.pi)) # M @ grid.phi for row-by-column product

    # apply LCG
    y, iter_conv = PrecondLinearConjGradPoisson(sigma_p, grid.indices7, x0=y) #riduce di 1/3 il numero di iterazioni necessarie a convergere
    
    # scale the field with the constrained 'force' term
    grid.phi = grid.phi - y / omega
    
    return grid, y, iter_conv


def PrecondLinearConjGradPoisson(b, index, x0 = np.zeros(N_tot), tol=tol):
    P_inv = - 1 / 6
    x = x0
    r = MatrixVectorProduct_7entries_1(x, index) - b
    
    v = P_inv * r 
    p = -v
    
    r_new = np.array(np.ones(N_tot))
    iter = 0

    while np.linalg.norm(r_new) > tol:
        iter = iter + 1
        Ap = MatrixVectorProduct_7entries_1(p, index) # A @ d for row-by-column product

        #Ap = A @ p
        alpha = np.dot(r, v) / np.dot(p, Ap)
        x = x + alpha * p
        
        r_new = r + alpha * Ap
        v_new =  P_inv * r_new  

        beta = np.dot(r_new, v_new) / np.dot(r, v)
        p = -v_new + beta * p
        
        r = r_new
        v = v_new

    return x, iter



# alternative function for matrix-vector product
def MatrixVectorProduct_7entries(M, v, index):
    v_matrix = v[index]
    result = np.einsum('ij,ij->i', M, v_matrix)

    return result

