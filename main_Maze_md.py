##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

import numpy as np
from grid import *
from input import md_variables, grid_setting, a0, output_settings
from indices import dict_indices_CoordTon
from verlet import VerletPoisson, PrecondLinearConjGradPoisson, VerletSolutePart1, VerletSolutePart2
import time
from tqdm import tqdm
#from spline_analysis import spline_analysis


begin_time = time.time()
start_initialization = time.time()

# set output files
if output_settings.print_field:
    from output_md import file_output_field

if output_settings.print_performance:
    from output_md import file_output_performance

if output_settings.print_solute:
    from output_md import file_output_solute

if output_settings.print_energy:
    from output_md import file_output_energy

if output_settings.print_temperature:
    from output_md import file_output_temperature
      

# get variables from input
delta = md_variables.delta
h = grid_setting.h
L = grid_setting.L
N = grid_setting.N
N_steps = md_variables.N_steps
stride = md_variables.stride
initialization = md_variables.initialization
omega = md_variables.omega
dt = md_variables.dt
preconditioning = md_variables.preconditioning
elec = md_variables.elec
V = 27.211386245988
tol = md_variables.tol

#file_output_phi = open(path + 'Phi_N' + str(N) + '.csv', 'r+')
#file_output_phi.write("n,phi\n")

# initialize grid by inserting particles in the system
grid = Grid()

# print relevant info
print('\nSimulation with N =', N, 'with N_steps =', N_steps, 'and tol =', md_variables.tol)
for i, particle in enumerate(grid.particles):
    print('Position particle', i + 1, '=', particle.pos)
    print('Velocity particle', i + 1, '=', particle.vel)
print('\nInitialization:',initialization, ' and preconditioning:', preconditioning)
print('\nParameters:\nh =', h, "\ndt =", dt, "\nstride =", stride, '\nL =', L, '\nomega =', omega)
print('\nPotential:', md_variables.potential)
print('\nElec:', elec, '\tNotElec: ', not_elec,'\n')
print('\nPrint solute:', output_settings.print_solute, '\tPrint field: ', output_settings.print_field, 
      '\nPrint energy:', output_settings.print_energy, '\tPrint temperature:', output_settings.print_temperature,
      '\tPrint performance:', output_settings.print_temperature,'\n')

################################ STEP 0 Verlet ##########################################
#########################################################################################
# GIA SETTATO CONDIZIONI INIZIALI, I.E. PRIMO STEP DI VELOCITY VERLET

q_tot = 0
#compute 8 nearest neighbors for any particle
for particle in grid.particles:
    particle.NearestNeigh()
    q_tot = q_tot + particle.charge

print('Total charge q = ',q_tot)

# set charges with the weight function
grid.SetCharges()

#initialize matrix indices
grid.indices7 = (DetIndices_7entries()).astype(int)

# initialize the electrostatic field with CG                  
if preconditioning == "Yes":
    grid.phi_prev, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, grid.indices7, tol=tol)

# AGGIUSTA
#grid.LinkedCellInit(grid.particles[0].r_cutoff)

for particle in grid.particles:
    if elec:
        particle.ComputeForce_FD(grid, prev=True)

if not_elec:
    grid.ComputeForceNotElecBasic()
    #grid.ComputeForceNotElecLC()
    
grid.Energy(iter=0, prev=True, print_energy=output_settings.print_energy)

################################ STEP 1 Verlet ##########################################
#########################################################################################

# Velocity Verlet for the solute
grid.particles = VerletSolutePart1(grid.particles)

# compute 8 nearest neighbors for any particle
for particle in grid.particles:
    particle.NearestNeigh()

# set charges with the weight function
grid.SetCharges()
    
if preconditioning == "Yes":
    grid.phi, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, grid.indices7, tol=tol, x0=grid.phi_prev)

grid = VerletSolutePart2(grid)
grid.Energy(iter=1, print_energy=output_settings.print_energy)

################################ FINE INIZIALIZZAZIONE ##########################################
#########################################################################################

X = np.arange(0, L, h)
j = int(grid.particles[0].pos[1] / h)
k = int(grid.particles[0].pos[2] / h)

end_initialization = time.time()
print("\nInitialization time: {:.2f} s \n".format(end_initialization - start_initialization))

steps_init = 0
y = np.zeros(N_tot)

######################################### Verlet ############################################
#############################################################################################

# iterate over the number of steps (i.e times I move the particle 1)
for i in tqdm(range(N_steps)):
    #print('Step = ', i, ' t elapsed from init =', time.time() - end_initialization)
    # move the particles 
    grid.particles = VerletSolutePart1(grid.particles)

    # compute 8 nearest neighbors for any particle
    for particle in grid.particles:
        particle.NearestNeigh()
   
    # set charges with the weight function
    grid.SetCharges()

    # apply Verlet algorithm
    start_Verlet = time.time()
    grid, y, iter_conv = VerletPoisson(grid, y=y)
    end_Verlet = time.time()

    grid = VerletSolutePart2(grid)
    grid.Energy(print_energy=output_settings.print_energy, iter=i + 2)

    if i % stride == 0 and i >= steps_init:
        
        if output_settings.print_solute:
            for p in range(grid.N_p):
                file_output_solute.write(str(grid.particles[p].charge) + ',' + str(i) + ',' + str(p+1) 
                                  + ',' + str(grid.particles[p].pos[0]) + ',' + str(grid.particles[p].pos[1]) + ',' + str(grid.particles[p].pos[2])
                                  + ','  + str(grid.particles[p].vel[0]) + ',' + str(grid.particles[p].vel[1]) + ',' + str(grid.particles[p].vel[2]) + '\n') 
                                  #+ ','  + str(grid.particles[p].force[0]) + ',' + str(grid.particles[p].force[1]) + ',' + str(grid.particles[p].force[2]) + '\n')
        
        if output_settings.print_performance:
            file_output_performance.write(str(i) + ',' + str(end_Verlet - start_Verlet) + ',' + str(iter_conv) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n")
        
        if output_settings.print_field:
            field_x_MaZe = np.array([grid.phi[dict_indices_CoordTon[tuple([l, j, k])]] for l in range(N)])
            for n in range(N):
                file_output_field.write(str(i) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')


# close output files
if output_settings.print_field:
    file_output_field.close()

if output_settings.print_performance:
    file_output_performance.close()

if output_settings.print_solute:
    file_output_solute.close()

if output_settings.print_energy:
    file_output_energy.close()

if output_settings.print_temperature:
    file_output_temperature.close()
      
end_time = time.time()
print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))

