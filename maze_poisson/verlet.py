import os
from math import exp, tanh

import numpy as np
from scipy.linalg import blas

from .c_api import c_conj_grad, c_daxpy, c_ddot, c_laplace
from .profiling import profile


def VerletSolutePart1(grid, dt=None, thermostat=False):
    if dt is None:
        dt = grid.dt
    N_p = grid.N_p
    kB = grid.kB
    L = grid.L

    particles = grid.particles

    if thermostat == True:
        mi_vi2 = [p.mass * np.dot(p.vel, p.vel) for p in particles]
        T_exp = np.sum(mi_vi2) / (3 * N_p * kB)
        lambda_scaling = np.sqrt(T / T_exp)

    for particle in particles:
        if thermostat == True:
            particle.vel = particle.vel * lambda_scaling
        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
        particle.pos = particle.pos + particle.vel * dt 
        particle.pos = particle.pos - L * np.floor(particle.pos / L)   
    return particles


def VerletSolutePart2(grid, dt=None, prev=False):
    not_elec = grid.not_elec
    elec = grid.elec

    grid.potential_notelec = 0
    if not_elec:
        grid.ComputeForceNotElecBasic()

    for particle in grid.particles:
        if elec:
            particle.ComputeForce_FD(grid, prev=prev)
        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
    return grid

### OVRVO ###
     
### Solves the equations for the O-block ###
def O_block(v, m, gamma, dt, kBT):
    c1 = exp(-gamma*dt)
    rnd = np.random.multivariate_normal(mean = (0.0, 0.0, 0.0), cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v_t_dt = np.sqrt(c1) * v + np.sqrt((1 - c1) * kBT / m) * rnd
    return v_t_dt

### Solves the equations for the V-block ###
def V_block(v, F, m, gamma, dt):
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2 /(gamma * dt) * tanh(0.5 * gamma * dt)) 
    v_t_dt = v + 0.5 * c2 * dt * F / m #F is the force
    return v_t_dt

### Solves the equations for the R-block ###
def R_block(x,v, gamma, dt, L):
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2 /(gamma * dt) * tanh(0.5 * gamma * dt)) 

    x_t_dt = x + c2 * dt * v
    x_t_dt = x_t_dt - L * np.floor(x_t_dt / L)   
    return x_t_dt

@profile
def OVRVO_part1(grid, thermostat=False):
    particles = grid.particles
    if thermostat:
        gamma_sim = grid.md_variables.gamma_inpt
    else:
        gamma_sim = 0

    dt = grid.md_variables.dt
    kBT = grid.md_variables.kBT
    L = grid.L
        
    for p in particles:
        p.vel = O_block(p.vel, p.mass, gamma_sim, dt, kBT)
        p.vel = V_block(p.vel,p.force + p.force_notelec, p.mass, gamma_sim, dt)
        p.pos = R_block(p.pos,p.vel, gamma_sim, dt, L)
    
    return particles

def OVRVO_part2(grid, prev=False, thermostat=False):
    dt = grid.md_variables.dt
    kBT = grid.md_variables.kBT

    if thermostat:
        gamma_sim = grid.md_variables.gamma_inpt
    else:
        gamma_sim = 0
        
    if grid.not_elec:
        grid.ComputeForceNotElecBasic()
            
    for p in grid.particles:
        if grid.elec:
            p.ComputeForce_FD(grid, prev=prev)
        p.vel = V_block(p.vel, p.force + p.force_notelec, p.mass, gamma_sim, dt)
        p.vel = O_block(p.vel, p.mass, gamma_sim, dt, kBT)
    
    return grid.particles

@profile
def MatrixVectorProduct_manual(v):
    # print('MatrixVectorProduct_manual')
    # print(v.flags)
    # exit()
    res = np.empty_like(v)
    c_laplace(v, res, v.shape[0])
    return res
    # res = -6 * np.copy(v)

    # res[1:,:,:] += v[:-1,:,:]
    # res[:-1,:,:] += v[1:,:,:]
    # res[-1,:,:] += v[0,:,:]
    # res[0,:,:] += v[-1,:,:]

    # res[:,1:,:] += v[:,:-1,:]
    # res[:,:-1,:] += v[:,1:,:]
    # res[:,-1,:] += v[:,0,:]
    # res[:,0,:] += v[:,-1,:]

    # res[:,:,1:] += v[:,:,:-1]
    # res[:,:,:-1] += v[:,:,1:]
    # res[:,:,-1] += v[:,:,0]
    # res[:,:,0] += v[:,:,-1]

    # return res

MatrixVectorProduct = MatrixVectorProduct_manual

# apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoisson(grid, y):
    omega = grid.md_variables.omega
    tol = grid.md_variables.tol
    h = grid.h

    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = tmp

    # compute the constraint with the provisional value of the field phi
    matrixmult = MatrixVectorProduct(grid.phi)
    sigma_p = omega * (grid.q / h + matrixmult / (4 * np.pi)) # M @ grid.phi for row-by-column product

    # apply LCG
    y_new, iter_conv = PrecondLinearConjGradPoisson(sigma_p, x0=y, tol=tol) #riduce di 1/3 il numero di iterazioni necessarie a convergere
    
    # scale the field with the constrained 'force' term
    grid.phi -= y_new / omega * (4 * np.pi)

    if grid.debug:
        matrixmult1 = MatrixVectorProduct(y_new)
        print('LCG precision     :',np.max(np.abs(matrixmult1 - sigma_p)))
        
        matrixmult1_old = MatrixVectorProduct(y_new / omega)
        print('LCG precision orig:',np.max(np.abs(matrixmult1_old - sigma_p / omega)))
    
        matrixmult2 = MatrixVectorProduct(grid.phi)
        sigma_p1 = grid.q / h + matrixmult2 / (4 * np.pi) # M @ grid.phi for row-by-column product
    
        print('max of constraint: ', np.max(np.abs(sigma_p1)),'\n')
    
    return grid, y_new, iter_conv

@profile
def PrecondLinearConjGradPoisson(b, x0 = None, tol=1e-7):
    x = np.empty_like(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    i = c_conj_grad(b, x0, x, tol, b.shape[0])
    return x, i
    # N_tot = b.size
    # if x0 is None:
    #     x0 = np.zeros_like(b)
    # P_inv = - 1 / 6
    # x = np.copy(x0)
    # r = MatrixVectorProduct(x) - b
    # v = P_inv * r  
    # p = -v
    
    # r_new = np.ones_like(r)
    # iter = 0

    # while np.linalg.norm(r_new) > tol:
    #     iter = iter + 1
    #     Ap = MatrixVectorProduct(p) # A @ d for row-by-column product
    #     # r_dot_v = blas.ddot(r, v, N_tot)
    #     # r_dot_v = np.sum(r * v)
    #     r_dot_v = c_ddot(r, v, N_tot)

    #     # alpha = r_dot_v / blas.ddot(p, Ap, N_tot)
    #     # alpha = r_dot_v / np.sum(p * Ap)
    #     alpha = r_dot_v / c_ddot(p, Ap, N_tot)
    #     # x = blas.daxpy(p, x , N_tot, alpha)
    #     # x = alpha * p + x
    #     c_daxpy(p, x, x, alpha, N_tot)
    #     # c_daxpy2(p, x, alpha, N_tot)
 
    #     # r_new = blas.daxpy(Ap, r, N_tot, alpha)
    #     # r_new = alpha * Ap + r
    #     c_daxpy(Ap, r, r_new, alpha, N_tot)
    #     # c_daxpy2(Ap, r, alpha, N_tot)
    #     # r_new = r
    #     v_new =  P_inv * r_new

    #     # beta = blas.ddot(r_new, v_new, N_tot) / r_dot_v
    #     # beta = np.sum(r_new * v_new) / r_dot_v
    #     beta = c_ddot(r_new, v_new, N_tot) / r_dot_v
    #     # p = blas.daxpy(p, -v_new, N_tot, beta)
    #     # p = beta * p - v_new
    #     c_daxpy(p, -v_new, p, beta, N_tot)
    #     # v_new = -v_new
    #     # c_daxpy2(p, v_new, beta, N_tot)
        
    #     r = r_new
    #     v = v_new

    # return x, iter

# alternative function for matrix-vector product
def MatrixVectorProduct_7entries(M, v, index):
    v_matrix = v[index]
    result = np.einsum('ij,ij->i', M, v_matrix)
    return result

# apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoissonBerendsen(grid,eta):
    omega = grid.md_variables.omega
    h = grid.h
    tol = grid.md_variables.tol

    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = np.copy(tmp)

    # apply SHAKE
    const_inv = 1 / 42
    stop_iteration =  False
    iter = 0

    # compute the constraint with the provisional value of the field phi
    M_phi = MatrixVectorProduct(grid.phi)
    sigma_p = grid.q / h + M_phi / (4 * np.pi) # M @ grid.phi for row-by-column product

    while(stop_iteration == False):	
        iter = iter + 1
        delta_eta =  -(4 * np.pi)**2 * const_inv * sigma_p * omega
        eta = eta + delta_eta
        
        M_delta_eta = MatrixVectorProduct(delta_eta)
        grid.phi = grid.phi + M_delta_eta / (4 * np.pi) 
        
        M_phi = MatrixVectorProduct(grid.phi)
        sigma_p = grid.q / h + M_phi / (4 * np.pi) # M @ grid.phi for row-by-column product
                
        if grid.output_settings.print_iters:
            # from .output_md import OutputFiles
            grid.output_files.file_output_iters.write(str(iter) + ',' + str(np.max(np.abs(sigma_p))) + ',' + str(np.linalg.norm(np.abs(sigma_p))) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n")
             
        # if np.linalg.norm(sigma_p) < tol: # MAX OR NORM?
        if np.max(np.abs(sigma_p)) < tol :
            stop_iteration = True
    
    print('iter=',iter)

    if grid.debug:
        matrixmult2 = MatrixVectorProduct(grid.phi)
        sigma_p1 = grid.q / h + matrixmult2 / (4 * np.pi) # M @ grid.phi for row-by-column product
    
        print('max of constraint: ', np.max(np.abs(sigma_p1)))
        print('norm of constraint: ', np.linalg.norm(sigma_p),'\n')
    return grid, eta, iter
