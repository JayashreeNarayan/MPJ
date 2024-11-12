##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

#from ..finite_difference import FiniteDifference
import time

import numpy as np
from tqdm import tqdm

from ..grid import *
from ..indices import dict_indices_CoordTon
from ..input import a0, grid_setting, md_variables
from ..output import (file_output_convergence, file_output_field,
                      file_output_solute, file_output_time)
from ..verlet import (LinearConjGrad, PrecondLinearConjGradPoisson,
                      VerletPoisson)

#import sys

def main():
    V = 27.211386245988
    begin_time = time.time()
    start_initialization = time.time()


    # get variables from input
    dx = md_variables.dx
    delta = md_variables.delta
    h = grid_setting.h
    L = grid_setting.L
    N = grid_setting.N
    N_steps = md_variables.N_steps
    stride = md_variables.stride
    initialization = md_variables.initialization
    dt = md_variables.dt
    preconditioning = md_variables.preconditioning
    tol = md_variables.tol
    omega = md_variables.omega

    # initialize grid by inserting particles in the system
    grid = Grid()
    print('\nSimulation with N =', N, 'with N_steps =', N_steps, 'and tol =', md_variables.tol)
    for i, particle in enumerate(grid.particles):
        print('Position particle', i + 1, '=', particle.pos)
    print('\nInitialization:',initialization, ' and preconditioning:', preconditioning)
    print('\nParameters:\nh =', h, "\ndt =", dt, "\nstride =", stride, "\ndx =", dx, '\nL =', L, '\nomega =', omega)

    #compute 8 nearest neighbors for any particle
    for particle in grid.particles:
        particle.NearestNeigh()
        #print(particle.pos)

    # set charges with the weight function
    grid.SetCharges()

    #initialize matrix indices
    grid.indices7 = (DetIndices_7entries()).astype(int)
    #M_init = np.zeros((N_tot, N_tot))
    #old_list1 = grid.particles[0].ComputeInitialNeighField()
    #old_list2 = grid.particles[1].ComputeInitialNeighField()
    #old_list = np.array([old_list1, old_list2])

    #start_Matrix = time.time()
    #M = InitializeMatrix_7EntriesPBC() 
    #end_Matrix = time.time()

    # initialize the electrostatic field with CG                  
    if initialization == 'CG':
        if preconditioning == "Yes":
            #grid.phi_prev = PrecondLinearConjGrad(M,- 4 * np.pi * grid.q / h, grid.indices7, tol=1e-5)
            grid.phi_prev = PrecondLinearConjGradPoisson(- omega * 4 * np.pi * grid.q / h, grid.indices7, tol=1e-5)
        #else:
        #    grid.phi_prev = LinearConjGrad(M, - 4 * np.pi * grid.q / h, grid.indices7, tol=tol)
        
        # move one of the particles away from the other
        for particle in grid.particles:
            #delta = np.zeros(3)
            #dir = np.random.randint(0,2)
            #delta[dir] = dx
            if particle.charge > 0:
                particle.Move(-delta)
            if particle.charge < 0:
                particle.Move(delta)
            #print(particle.pos)
        
        # compute 8 nearest neighbors for any particle
        for particle in grid.particles:
            particle.NearestNeigh()


        # set charges with the weight function
        grid.SetCharges()
        
        # update matrix
        #M = UpdateMatrix_7EntriesPBC(grid, M) 
        #M = UpdateProvaMatrix_7EntriesPBC(grid, M) 
        
        if preconditioning == "Yes":
            #grid.phi = PrecondLinearConjGrad(M, - 4 * np.pi * grid.q / h, grid.indices7, tol=1e-5, x0=grid.phi_prev)
            grid.phi = PrecondLinearConjGradPoisson(- omega * 4 * np.pi * grid.q / h, grid.indices7, tol=1e-5, x0=grid.phi_prev)
        #else:
        #    grid.phi = LinearConjGrad(M, - 4 * np.pi * grid.q / h, grid.indices7, tol=tol, x0=grid.phi_prev)
        

    # plot the step 0 and 1 field
    X = np.arange(0, L, h)
    j = int(grid.particles[0].pos[1] / h)
    k = int(grid.particles[0].pos[2] / h)


    end_initialization = time.time()
    print("\nInitialization time: {:.2f} s \n".format(end_initialization - start_initialization))

    y = np.zeros(N_tot)

    # iterate over the number of steps (i.e times I move the particle 1)
    for i in tqdm(range(N_steps)):
        # move one of the particles away from the other
        for particle in grid.particles:
            #delta = np.zeros(3)
            #dir = np.random.randint(0,2)
            #delta[dir] = dx
            if particle.charge > 0:
                particle.Move(-delta)
            if particle.charge < 0:
                particle.Move(delta)
        
        print('distance x =', (grid.particles[0].pos - grid.particles[1].pos)[0])
        # compute 8 nearest neighbors for any particle
        for particle in grid.particles:
            particle.NearestNeigh()
    
        # set charges with the weight function
        start_charges = time.time()
        grid.SetCharges()
        end_charges = time.time()

        # apply Verlet algorithm
        start_Verlet = time.time()
        grid, y = VerletPoisson(grid, y=y)
        end_Verlet = time.time()
        
        E_tot = 0
        for n_p, particle in enumerate(grid.particles):
            E_tot = E_tot + particle.Energy(grid.phi, grid.q)
            #print('energy particle ', n_p + 1, '=', )
        print('E_tot =', E_tot)

        # print results
        if i % stride == 0:
            for p in range(grid.N_p):
                file_output_solute.write(str(grid.particles[p].charge) + ',' + str(i) + ',' + str(p+1) 
                                    + ',' + str(grid.particles[p].pos[0]) + ',' + str(grid.particles[p].pos[1]) + ',' + str(grid.particles[p].pos[2])
                                    + ','  + str(grid.particles[p].vel[0]) + ',' + str(grid.particles[p].vel[1]) + ',' + str(grid.particles[p].vel[2]) 
                                    + ','  + str(grid.particles[p].force[0]) + ',' + str(grid.particles[p].force[1]) + ',' + str(grid.particles[p].force[2]) + '\n')
            
            file_output_time.write(str(i) + ',' + str(end_Verlet - start_Verlet) + ',' + str(end_charges - start_charges) + "\n") #+ "\n") #
            field_x_MaZe = np.array([grid.phi[dict_indices_CoordTon[tuple([l, j, k])]] for l in range(N)])
        
            for n in range(N):
                #file_output_field.write(str(N) + ',' + str(i) + ',' + str(X[n]) + ',' + str(field_x_MaZe[n]) + ',' + str(field_x_fd[n]) + ',' + str(field_x_th[n]) + '\n')
                file_output_field.write(str(N) + ',' + str(i) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')

    # close output files
    file_output_field.close()
    file_output_time.close()
    file_output_convergence.close()
    file_output_solute.close()
    end_time = time.time()
    print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))



    #file_output_field.write(str(N) + ',' + str(i) + ',' + str(X[n]) + ',' + str(field_x_MaZe[n]) + ',' + str(field_x_th[n]) + '\n')
    #print(str(N) + '\t' + str(i + 1) + '\t' + str(X[j]) + '\t' + str(field_x_MaZe[j]) + '\t' + str(field_x_fd[j]) + '\t' + str(field_x_th[j]) + '\n')
    #plot_field_comparison3(X, field_x_MaZe, field_x_th, field_x_fd, step = i + 1)
    #plot_field_comparison(X, field_x_MaZe, field_x_th)
    #plot_field(X, field_x_MaZe)

    #plot_field_comparison(X, field_x_MaZe, field_x_th)
    #field_x_prev = [grid.phi_prev[nFromIndices([i, j, k])] for i in range(N)]
    #plot_field_comparison3(X, field_x_MaZe, field_x_th, field_x_fd)
    #plot_field_comparison2_steps(X, field_x_MaZe, field_x_th)
    #plot_discrepancy3(X, field_x_th, field_x_MaZe, field_x_fd)
    #plot_field(X, field_x, prev=True)
    #plot_field(X, field_x_th, prev=True)
