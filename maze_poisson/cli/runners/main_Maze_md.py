##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

import time

import numpy as np
from tqdm import tqdm

from ...constants import a0, t_au
from ...grid import *
from ...restart import generate_restart
from ...verlet import (OVRVO_part1, OVRVO_part2, PrecondLinearConjGradPoisson,
                       VerletPoisson, VerletSolutePart1, VerletSolutePart2)


def main(grid_setting, output_settings, md_variables):
    begin_time = time.time()
    start_initialization = time.time()

    # get variables from input
    h = grid_setting.h
    L = grid_setting.L
    L_ang = grid_setting.L_ang
    N = grid_setting.N
    N_p = grid_setting.N_p
    h_ang = L_ang/N

    T = md_variables.T
    not_elec = md_variables.not_elec
    N_steps = md_variables.N_steps
    stride = md_variables.stride
    initialization = md_variables.initialization
    thermostat = md_variables.thermostat
    dt = md_variables.dt
    dt_fs = md_variables.dt_fs
    preconditioning = md_variables.preconditioning
    rescale = md_variables.rescale
    elec = md_variables.elec
    V = 27.211386245988
    tol = md_variables.tol
    iter_restart = output_settings.iter_restart

    # initialize grid by inserting particles in the system
    grid = Grid(grid_setting, md_variables, output_settings)

    ofiles = grid.output_files

    # print relevant info
    print('\nSimulation with N =', N, 'with N_steps =', N_steps, 'and tol =', md_variables.tol)
    #for i, particle in enumerate(grid.particles):
    #    print('Position particle', i + 1, '=', particle.pos)
    #    print('Velocity particle', i + 1, '=', particle.vel)
    print('\nInitialization is done with CG and preconditioning:', preconditioning)
    print('\nParameters:\nh =', h_ang, 'A', "\ndt =", dt_fs, 'fs', "\nstride =", stride, '\nL =', L_ang, 'A', '\ngamma =', md_variables.gamma)
    print('\nPotential:', md_variables.potential)
    print('\nElec:', elec, '\tNotElec: ', not_elec,'\n')
    print('\nTemperature:',T,' K\tNumerical density:',N_p / L**3,' a.u.')
    print('\nPrint solute:', output_settings.print_solute, '\tPrint field: ', output_settings.print_field, '\tPrint tot_force:', output_settings.print_tot_force,
        '\nPrint energy:', output_settings.print_energy, '\tPrint temperature:', output_settings.print_temperature,
        '\tPrint performance:', output_settings.print_performance,'\tRestart:', output_settings.restart,'\n')
    print('\nThermostat:', thermostat, '\tRescaling of the velocities:', rescale)

    ################################ STEP 0 Verlet ##########################################
    #########################################################################################

    q_tot = 0
    #compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    q_tot = np.sum(grid.particles.charges)
    print('Total charge q = ',q_tot)

    # set charges with the weight function
    grid.SetCharges()

    # initialize the electrostatic field with CG                  
    if preconditioning == "Yes":
        grid.phi_prev, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol)

    # AGGIUSTA
    #grid.LinkedCellInit(grid.particles[0].r_cutoff)

    if not_elec:
        grid.particles.ComputeForceNotElec()

    if elec:
        grid.particles.ComputeForce_FD(prev=True) 

    ################################ STEP 1 Verlet ##########################################
    #########################################################################################

    # Velocity Verlet for the solute
    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part1(grid, thermostat = thermostat)
    else:
        grid.particles = VerletSolutePart1(grid, thermostat=thermostat)

    # compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()

    # set charges with the weight function
    grid.SetCharges()
        
    if preconditioning == "Yes":
        grid.phi, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol, x0=grid.phi_prev)

    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part2(grid, thermostat = thermostat)
    else:
        grid = VerletSolutePart2(grid)

    # rescaling of the velocities to get total momentum = 0
    if rescale:
        print('\nRescaling of the velocities in progress...')
        grid.RescaleVelocities()

    ################################ FINE INIZIALIZZAZIONE ##########################################
    #########################################################################################

    X = np.arange(0, L, h)
    j = int(grid.particles.pos[0,1] / h)
    k = int(grid.particles.pos[0,2] / h)

    end_initialization = time.time()
    print("\nInitialization time: {:.2f} s \n".format(end_initialization - start_initialization))

    if output_settings.restart == True and thermostat == False:
        init_steps = 0
    else:
        init_steps = md_variables.init_steps
        
    print('Number of initialization steps:', init_steps,'\n')

    y = np.zeros_like(grid.q) 

    ######################################### Verlet ############################################
    #############################################################################################

    counter = 0 
  
    # iterate over the number of steps (i.e times I move the particle 1)
    for i in tqdm(range(N_steps)):
        #print('\nStep = ', i, ' t elapsed from init =', time.time() - end_initialization)
        if md_variables.integrator == 'OVRVO':
            grid.particles = OVRVO_part1(grid, thermostat = thermostat)
        else:
            grid.particles = VerletSolutePart1(grid, thermostat = thermostat)

        if elec:
            # compute 8 nearest neighbors for any particle
            grid.particles.NearestNeighbors()
        
            # set charges with the weight function
            grid.SetCharges()

            # apply Verlet algorithm
            start_Verlet = time.time()
            grid, y, iter_conv = VerletPoisson(grid, y=y)
            #grid, y, iter_conv = VerletPoissonBerendsen(grid, y)
            end_Verlet = time.time()

        if md_variables.integrator == 'OVRVO':
            grid.particles = OVRVO_part2(grid, thermostat = thermostat)
        else:
            grid = VerletSolutePart2(grid)

        if output_settings.print_tot_force:
            tot_force = np.zeros(3)
            tot_force = np.sum(grid.particles.forces + grid.particles.forces_notelec, axis=0)
            
            ofiles.file_output_tot_force.write(str(i) + ',' + str(tot_force[0]) + ',' + str(tot_force[1]) + ','+ str(tot_force[2]) + "\n") 
            ofiles.file_output_tot_force.flush()

        grid.Energy(print_energy=output_settings.print_energy, iter=i)
        grid.Temperature(print_temperature=output_settings.print_temperature, iter=i)
        
                
        if i % stride == 0 and i >= init_steps:
            if counter == 0 and thermostat == True:
                print('End of thermostatting')
                thermostat = False
                counter = counter + 1
            
            if output_settings.print_solute:
                for p in range(grid.N_p):
                    ofiles.file_output_solute.write(str(grid.particles.charges[p]) + ',' + str(i - init_steps) + ',' + str(p) 
                                    + ',' + str(grid.particles.pos[p, 0]) + ',' + str(grid.particles.pos[p, 1]) + ',' + str(grid.particles.pos[p, 2]) # pos are printed in a.u., then converted to angs in the convert_to_xyz file
                                    + ','  + str(grid.particles.vel[p, 0]) + ',' + str(grid.particles.vel[p, 1]) + ',' + str(grid.particles.vel[p, 2])  #vel are in a.u.
                                    + ','  + str(grid.particles.forces[p, 0]) + ',' + str(grid.particles.forces[p, 1]) + ',' + str(grid.particles.forces[p, 2]) + '\n')
                                    #+ ','  + str(grid.particles[p].force_notelec[0]) + ',' + str(grid.particles[p].force_notelec[1]) + ',' + str(grid.particles[p].force_notelec[2]) + '\n')
                                    #+ ','  + str(grid.particles[p].force[0] + grid.particles[p].force_notelec[0]) + ',' + str(grid.particles[p].force[1]+ grid.particles[p].force_notelec[1]) + ',' + str(grid.particles[p].force[2]+ grid.particles[p].force_notelec[2]) + '\n')          
            
            if output_settings.print_performance and elec:
                ofiles.file_output_performance.write(str(i - init_steps) + ',' + str(end_Verlet - start_Verlet) + ',' + str(iter_conv) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n"
                        
            if output_settings.print_field and elec:
                field_x_MaZe = np.array([grid.phi[l, j, k] for l in range(N)])
                for n in range(N):
                    ofiles.file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')

    # close output files
    if output_settings.print_field:
        ofiles.file_output_field.close()

    if output_settings.print_performance:
        ofiles.file_output_performance.close()

    #if output_settings.print_iters:
    #    file_output_iters.close()

    if output_settings.print_solute:
        ofiles.file_output_solute.close()

    if output_settings.print_energy:
        ofiles.file_output_energy.close()

    if output_settings.print_temperature:
        ofiles.file_output_temperature.close()

    if output_settings.generate_restart_file:
        restart_file = generate_restart(grid_setting, output_settings, iter_restart)
        print('Restart file generated: ', restart_file)
        
    end_time = time.time()
    print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))
