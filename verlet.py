from input import grid_setting, md_variables
import numpy as np
import time 
#from output import file_output_convergence
from output_md import file_output_convergence  #BE CAREFUL
from scipy.sparse import linalg
from particle import g, BoxScale 


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


def VerletSolutePart2(grid, dt=dt, prev=False):
    grid.potential_notelec = 0
    if not_elec:
        #grid.ComputeForceNotEleLC() #TF or LJ
        grid.ComputeForceNotElecBasic()
    
    tot_force = 0
    for particle in grid.particles:
 
        if elec:
            #particle.ComputeForce(grid, prev=prev)
            #particle.ComputeForce_CubicSpline(grid, prev=prev)
            particle.ComputeForce_FD(grid, prev=prev)
            #
            #particle.ComputeForce_ParticlePos(grid, prev=prev)

        particle.vel = particle.vel + 0.5 * ((particle.force + particle.force_notelec) / particle.mass) * dt
        tot_force = tot_force + particle.force + particle.force_notelec
    
    #print('tot force = ', tot_force)
    return grid


def MatrixVectorProduct_7entries(M, v, index):
    v_matrix = v[index]
    result = np.einsum('ij,ij->i', M, v_matrix)

    return result


def MatrixVectorProduct_7entries_1(v, index):
    M = np.array([1,1,1,-6,1,1,1])    
    v_matrix = v[index]
    result = np.dot(v_matrix,M)

    return result


#to compute condition number: np.linalg.cond()
def Jacobi():
    return - 1 / 6
'''
# incomplete Poisson preconditioning
def IPP():
    grid_shape = (N, N, N)
    A = linalg.LaplacianNd(grid_shape,boundary_conditions='periodic').tosparse()
    #print(A.toarray())
    #L = np.tril(A, k=-1)
    #diag_A = A.diagonal()
    LU = linalg.spilu(A)
    LU_operator = linalg.aslinearoperator(LU)
    print(LU_operator)
    return LU_operator

#IPP()
'''

#apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoisson(grid,y):
    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = np.copy(tmp)

    # compute the constraint with the provisional value of the field phi
    #beg1 = time.time()
    #matrixmult = MatrixVectorProduct_7entries(M, grid.phi, grid.indices7)
    #end1 = time.time()
    #beg2 = time.time()
    matrixmult = MatrixVectorProduct_7entries_1(grid.phi, grid.indices7)
    #end2 = time.time()
    #print(matrixmult - matrixmult1)
    #print('matmul = ', end1 - beg1, 'matmul1 = ', end2 - beg2)
    #counter = 0

    sigma_p = omega * (grid.q / h + matrixmult / (4 * np.pi)) # M @ grid.phi for row-by-column product
    #print('mean constraint =',np.mean(sigma_p))
    # apply LCG

    #y = PrecondLinearConjGrad(M,sigma_p, grid.indices7, x0=y) #riduce di 1/3 il numero di iterazioni necessarie a convergere
    
    y = PrecondLinearConjGradPoisson(sigma_p, grid.indices7, x0=y) #riduce di 1/3 il numero di iterazioni necessarie a convergere
 
    # scale the field with the constrained 'force' term
    grid.phi = grid.phi - y / omega
    
    return grid, y


#apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def Verlet(M, grid, y):
    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = np.copy(tmp)

    # compute the constraint with the provisional value of the field phi
    matrixmult = MatrixVectorProduct_7entries(M, grid.phi, grid.indices7)

    #counter = 0
    sigma_p = omega * (grid.q / h + matrixmult / (4 * np.pi)) # M @ grid.phi for row-by-column product

    # apply LCG
 
    if preconditioning == 'Yes':
        y = PrecondLinearConjGrad(M,sigma_p, grid.indices7, x0=y) #riduce di 1/3 il numero di iterazioni necessarie a convergere
    else:
        y = LinearConjGrad(M, sigma_p, grid.indices7, x0=y)
        
    # scale the field with the constrained 'force' term
    grid.phi = grid.phi - y 
    
    return grid, y

   
# Linear Conjugate Gradient function, to solve the system Ax = b
def LinearConjGrad(A, b, index, x0 = np.zeros(N_tot), tol=tol):
    x = x0
    r = b - MatrixVectorProduct_7entries(A, x, index)
    
    d = r
    r_new = np.array(np.ones(N_tot))
    iter = 0

    while np.linalg.norm(r_new) > tol:
        iter = iter + 1
        Ad = MatrixVectorProduct_7entries(A,d, index) # A @ d for row-by-column product

        #Ad = A @ d
        alpha = np.dot(r, r) / np.dot(d, Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad
    
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        d = r_new + beta * d
        r = r_new

    print("LCG: convergence reached at iter = ", iter,' tol = ',  np.linalg.norm(r_new) )
    #file_output_convergence.write(str(iter) + ',' + str(np.linalg.norm(r_new)) + "\n")   
    return x

# Linear Conjugate Gradient function, to solve the system Ax = b
def PrecondLinearConjGrad(A, b, index, x0 = np.zeros(N_tot), tol=tol):
    x = x0
    r = MatrixVectorProduct_7entries(A, x, index) - b
    
    v = r / A[:, 3]
    p = -v
    
    r_new = np.array(np.ones(N_tot))
    iter = 0

    while np.linalg.norm(r_new) > tol:
        iter = iter + 1
        Ap = MatrixVectorProduct_7entries(A,p, index) # A @ d for row-by-column product

        #Ap = A @ p
        alpha = np.dot(r, v) / np.dot(p, Ap)
        x = x + alpha * p
        
        r_new = r + alpha * Ap
        v_new = r_new / A[:, 3]

        beta = np.dot(r_new, v_new) / np.dot(r, v)
        p = -v_new + beta * p
        
        r = r_new
        v = v_new

    #print("Preconditioned LCG: convergence reached at iter = ", iter,' tol = ',  np.linalg.norm(r_new) )
    file_output_convergence.write(str(iter) + ',' + str(np.linalg.norm(r_new)) + "\n")   
    return x




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

    #print("Preconditioned LCG: convergence reached at iter = ", iter,' tol = ',  np.linalg.norm(r_new) )
    file_output_convergence.write(str(iter) + ',' + str(np.linalg.norm(r_new)) + "\n")   
    return x




# Linear Conjugate Gradient function, to solve the system (A1 + A2)x = b
'''
def LinearConjGradSplit(A1, A2, b, x0 = np.zeros(N_tot), tol=tol, max_iter=500):
    x = x0
    r = b - matmul_toeplitz(A1, x) - A2 * x
    d = r

    for i in range(max_iter):
        Ad = matmul_toeplitz(A1,d)  + A2 * d # A @ d for row-by-column product
        alpha = np.dot(r, r) / np.dot(d, Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad

        if np.linalg.norm(r_new) < tol:
            #print("LCG: convergence reached at iter = ", i)
            break
        elif np.linalg.norm(r_new) > tol and iter == (max_iter - 1):
            print("LCG: convergence NOT reached")
            
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        d = r_new + beta * d
        r = r_new
    return x

# Linear Conjugate Gradient function, to solve the system (A1 + A2)x = b
def LinearConjGradSplit2(A1, A2, A3, b, n_diff_zero, x0 = np.zeros(N_tot), tol=tol, max_iter=500):
    A3d = np.zeros(N_tot)
    A3x = np.zeros(N_tot)

    x = x0
    A3x_red = A3 @ x
    print(np.size(A3x_red))
    for j, n in enumerate(n_diff_zero):
        A3x[n] = A3x_red[j] 
        
    r = b - matmul_toeplitz(A1, x) - A2 * x - A3x
    d = r

    for i in range(max_iter):
        A3d_red = A3 @ d
        for j, n in enumerate(n_diff_zero):
            A3d[n] = A3d_red[j]
        
        Ad = matmul_toeplitz(A1,d)  + A2 * d + A3d# A @ d for row-by-column product
        alpha = np.dot(r, r) / np.dot(d, Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad

        if np.linalg.norm(r_new) < tol:
            #print("LCG: convergence reached at iter = ", i)
            break
        elif np.linalg.norm(r_new) > tol and iter == (max_iter - 1):
            print("LCG: convergence NOT reached")
            
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        d = r_new + beta * d
        r = r_new
    return x

# Linear Conjugate Gradient function, to solve the system (A1 + A2)x = b
def LinearConjGradSplit2Entire(A1, A2, A3, b, n_diff_zero, x0 = np.zeros(N_tot), tol=tol, max_iter=500):
    #A3d = np.zeros(N_tot)
    #A3x = np.zeros(N_tot)

    x = x0
    r = b - matmul_toeplitz(A1, x) - A2 * x - A3 @ x
    d = r

    for i in range(max_iter):
        Ad = matmul_toeplitz(A1,d)  + A2 * d + A3 @ d# A @ d for row-by-column product
        alpha = np.dot(r, r) / np.dot(d, Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad

        if np.linalg.norm(r_new) < tol:
            #print("LCG: convergence reached at iter = ", i)
            break
        elif np.linalg.norm(r_new) > tol and iter == (max_iter - 1):
            print("LCG: convergence NOT reached")
            
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        d = r_new + beta * d
        r = r_new
    return x

# check that the constraint is indeed satisfied for a certain timestep
def ComputeConstraint(grid, M, step = 1, print = False):
    if step == 0:
        sigma_p = grid.q / h + M @ grid.phi_prev / (4 * np.pi)
    elif step == 1:
        sigma_p = grid.q / h + M @ grid.phi / (4 * np.pi)
    if print == True:
        print(sigma_p)
    return sigma_p;
'''

#r = np.array([1,6,8])
#A = np.array([[1,2,3], [3,4,5], [6,7,8]])

#v = r / A[:,2]
#print(v[0], r[0]/A[0][2])
#print(v[1], r[1]/A[1][2])
#print(v[2], r[2]/A[2][2])
#print(v)