from input import grid_setting, md_variables, output_settings
import numpy as np
import time 
from math import exp, tanh
from concurrent.futures import ThreadPoolExecutor
debug = output_settings.debug

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
kBT = md_variables.kBT
gamma_inpt = md_variables.gamma


if output_settings.print_iters:
    from output_md import file_output_iters

def VerletSolutePart1(particles, dt=dt, thermostat=False):
    if thermostat == True:
        mi_vi2 = [p.mass * np.dot(p.vel, p.vel) for p in particles]
        T_exp = np.sum(mi_vi2) / (3 * N_p * kB)
        #print('T measured =', T_exp)
        lambda_scaling = np.sqrt(T / T_exp)

    for particle in particles:
        if thermostat == True:
            particle.vel = particle.vel * lambda_scaling
        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
        particle.pos = particle.pos + particle.vel * dt 
        particle.pos = particle.pos - L * np.floor(particle.pos / L)   
    return particles


def VerletSolutePart2(grid, dt=dt, prev=False):
    grid.potential_notelec = 0
    if not_elec:
        grid.ComputeForceNotElecBasic()

    for particle in grid.particles:
        if elec:
            particle.ComputeForce_FD(grid, prev=prev)
        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
        # berendsen
    return grid

### OVRVO ###
     
### Solves the equations for the O-block ###
def O_block(v, m, gamma):
    c1 = exp(-gamma*dt)
    rnd = np.random.multivariate_normal(mean = (0.0, 0.0, 0.0), cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    v_t_dt = np.sqrt(c1) * v + np.sqrt((1 - c1) * kBT / m) * rnd
    return v_t_dt

### Solves the equations for the V-block ###
def V_block(v, F, m, gamma):
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2 /(gamma * dt) * tanh(0.5 * gamma * dt)) 
    
    v_t_dt = v + 0.5 * c2 * dt * F / m #F is the force
    return v_t_dt

### Solves the equations for the R-block ###
def R_block(x,v, gamma):
    #print(gamma)
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2 /(gamma * dt) * tanh(0.5 * gamma * dt)) 

    x_t_dt = x + c2 * dt * v
    x_t_dt = x_t_dt - L * np.floor(x_t_dt / L)   
    return x_t_dt

def OVRVO_part1(particles, thermostat=False):
    if thermostat:
        gamma_sim = gamma_inpt
    else:
        gamma_sim = 0
        
    for p in particles:
        p.vel = O_block(p.vel, p.mass, gamma_sim)
        p.vel = V_block(p.vel,p.force + p.force_notelec, p.mass, gamma_sim)
        p.pos = R_block(p.pos,p.vel, gamma_sim)
    
    return particles

def OVRVO_part2(grid, prev=False, thermostat=False):
    if thermostat:
        gamma_sim = gamma_inpt
    else:
        gamma_sim = 0
        
    if not_elec:
        #grid.ComputeForceNotEleLC() #TF or LJ
        grid.ComputeForceNotElecBasic()
            
    for p in grid.particles:
        if elec:
            p.ComputeForce_FD(grid, prev=prev)
        p.vel = V_block(p.vel, p.force + p.force_notelec, p.mass, gamma_sim)
        p.vel = O_block(p.vel, p.mass, gamma_sim)
    
    return grid.particles

def MatrixVectorProduct(v): # added by davide
    app = np.copy(v).reshape((N, N, N))

    res = np.zeros((N, N, N))

    res -= app * 6
    for idx in (-1, 1):
        for ax in range(3):
            res += np.roll(app, idx, axis=ax)

    return res.flatten()

#apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoisson(grid,y):
    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = np.copy(tmp)

    # compute the constraint with the provisional value of the field phi
    matrixmult = MatrixVectorProduct(grid.phi)
    sigma_p = omega * (grid.q / h + matrixmult / (4 * np.pi)) # M @ grid.phi for row-by-column product

    # apply LCG
    y_new, iter_conv = PrecondLinearConjGradPoisson(sigma_p, x0=y) #riduce di 1/3 il numero di iterazioni necessarie a convergere
    
    # scale the field with the constrained 'force' term
    grid.phi = grid.phi - y_new / omega * (4 * np.pi)

    if debug:
        matrixmult1 = MatrixVectorProduct(y_new)
        print('LCG precision     :',np.max(np.abs(matrixmult1 - sigma_p)))
        
        matrixmult1_old = MatrixVectorProduct(y_new / omega)
        print('LCG precision orig:',np.max(np.abs(matrixmult1_old - sigma_p / omega)))
    
        matrixmult2 = MatrixVectorProduct(grid.phi)
        sigma_p1 = grid.q / h + matrixmult2 / (4 * np.pi) # M @ grid.phi for row-by-column product
    
        print('max of constraint: ', np.max(np.abs(sigma_p1)),'\n')
    
    return grid, y_new, iter_conv


def PrecondLinearConjGradPoisson(b, x0 = np.zeros(N_tot), tol=tol):
    P_inv = - 1 / 6
    x = x0
    r = MatrixVectorProduct(x) - b
    
    v = P_inv * r  # same as y in book
    p = -v
    
    r_new = np.array(np.ones(N_tot))
    iter = 0

    while np.linalg.norm(r_new) > tol:
        iter = iter + 1
        Ap = MatrixVectorProduct(p) # A @ d for row-by-column product

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


#apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoissonBerendsen(grid,eta):
    omega = md_variables.omega
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
        #delta_eta =  -(4 * np.pi) * const_inv * sigma_p * omega
        eta = eta + delta_eta
        
        M_delta_eta = MatrixVectorProduct(delta_eta)
        grid.phi = grid.phi + M_delta_eta / (4 * np.pi) 
        #grid.phi = grid.phi + M_delta_eta
        
        M_phi = MatrixVectorProduct(grid.phi)
        sigma_p = grid.q / h + M_phi / (4 * np.pi) # M @ grid.phi for row-by-column product
        #print(iter, np.max(np.abs(sigma_p)))
                
        if output_settings.print_iters:
            file_output_iters.write(str(iter) + ',' + str(np.max(np.abs(sigma_p))) + ',' + str(np.linalg.norm(np.abs(sigma_p))) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n")
             
        #if np.linalg.norm(sigma_p) < tol: # MAX OR NORM?
        if np.max(np.abs(sigma_p)) < tol :
            stop_iteration = True

        #if iter % 1000 == 0:
        #    omega = omega - 0.1
            
    
    print('iter=',iter)

    if debug:
        matrixmult2 = MatrixVectorProduct(grid.phi)
        sigma_p1 = grid.q / h + matrixmult2 / (4 * np.pi) # M @ grid.phi for row-by-column product
    
        print('max of constraint: ', np.max(np.abs(sigma_p1)))
        print('norm of constraint: ', np.linalg.norm(sigma_p),'\n')
    return grid, eta, iter
