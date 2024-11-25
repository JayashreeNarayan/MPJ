##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

import time
from loggers import logger_func
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...constants import a0, t_au
from ...grid import *
from ...restart import generate_restart
from ...verlet import (OVRVO_part1, OVRVO_part2, PrecondLinearConjGradPoisson,
                       VerletPoisson, VerletSolutePart1, VerletSolutePart2)

def main(grid_setting, output_settings, md_variables):
    begin_time = time.time()
    start_initialization = time.time()
    logger_func('--------Initialization begins---------', 'i')

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
    print('\nInitialization is done with CG and preconditioning:', preconditioning)
    print('\nParameters:\nh =', h_ang, 'A', "\ndt =", dt_fs, 'fs', "\nstride =", stride, '\nL =', L_ang, 'A', '\ngamma =', md_variables.gamma)
    print('\nPotential:', md_variables.potential)
    print('\nElec:', elec, '\tNotElec: ', not_elec,'\n')
    print('\nTemperature:',T,' K\tNumerical density:',N_p / L**3,' a.u.')
    print('\nPrint solute:', output_settings.print_solute, '\tPrint field: ', output_settings.print_field, '\tPrint tot_force:', output_settings.print_tot_force,
        '\nPrint energy:', output_settings.print_energy, '\tPrint temperature:', output_settings.print_temperature,
        '\tPrint performance:', output_settings.print_performance,'\tRestart:', output_settings.restart,'\n')
    print('\nThermostat:', thermostat, '\tRescaling of the velocities:', rescale)

    # log all the relevant info 
    logger_func('Simulation with N ='+str(N)+' with N_steps ='+str(N_steps)+' and tol ='+str(md_variables.tol)+'\nInitialization is done with CG and preconditioning: '+str(preconditioning)+ 
                '\nParameters: h ='+str(h_ang,)+ ' A \ndt ='+str(dt_fs)+' fs'+"\nstride ="+str(stride)+'\nL ='+str(L_ang)+' A'+'\ngamma ='+ str(md_variables.gamma)+
                '\nPotential:'+ str(md_variables.potential)+ 
                '\nElec:'+str(elec)+ '\tNotElec: '+str(not_elec)+'\n'+'\nTemperature:'+str(T)+' K\tNumerical density:'+str(N_p / L**3)+' a.u.'+
                '\nPrint solute:'+str(output_settings.print_solute)+ '\tPrint field: '+str(output_settings.print_field)+'\tPrint tot_force:'+str(output_settings.print_tot_force)+
                '\nPrint energy:'+str(output_settings.print_energy)+ '\tPrint temperature:'+str(output_settings.print_temperature)+
                '\tPrint performance:'+str(output_settings.print_performance)+'\tRestart:'+str(output_settings.restart)+'\n'+
                '\nThermostat:'+str(thermostat)+ '\tRescaling of the velocities:'+str(rescale), 'i')

    ################################ STEP 0 Verlet ##########################################
    #########################################################################################

    q_tot = 0
    #compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    q_tot = np.sum(grid.particles.charges)
    print('Total charge q = ',q_tot)
    logger_func('Total charge q = '+str(q_tot), 'i')

    # set charges with the weight function
    grid.SetCharges()
    #logger_func('Charges are set', 'i')

    # initialize the electrostatic field with CG                  
    if preconditioning == "Yes":
        #logger_func('Preconditioning being done for elec field', 'i')
        grid.phi_prev, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol)

    if not_elec:
        grid.particles.ComputeForceNotElec()
        #logger_func('Non_elec force being computed', 'i')

    if elec:
        grid.particles.ComputeForce_FD(prev=True) 
        #logger_func('Elec force being computed', 'i')
    
    else:
        logger_func("There is an error here", 'e')

    ################################ STEP 1 Verlet ##########################################
    #########################################################################################

    # Velocity Verlet for the solute
    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part1(grid, thermostat = thermostat)
        #logger_func('Thermostat being applied', 'i')
    else:
        grid.particles = VerletSolutePart1(grid, thermostat=thermostat)
        #logger_func('Thermostat is not applied', 'i')

    # compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    #logger_func('Nearest neighbours calculated', 'i')

    # set charges with the weight function
    grid.SetCharges()
    #logger_func("Charges set with weight function", 'i')
        
    if preconditioning == "Yes":
        grid.phi, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol, x0=grid.phi_prev)

    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part2(grid, thermostat = thermostat)
        #logger_func('OVRVO part 2 being run', 'i')
    else:
        grid = VerletSolutePart2(grid)

    # rescaling of the velocities to get total momentum = 0
    if rescale:
        print('\nRescaling of the velocities in progress...')
        logger_func('Rescaling of the velocities in progress...', 'i')
        grid.RescaleVelocities()

    ################################ FINE INIZIALIZZAZIONE ##########################################
    #########################################################################################

    X = np.arange(0, L, h)
    j = int(grid.particles.pos[0,1] / h)
    k = int(grid.particles.pos[0,2] / h)

    end_initialization = time.time()
    print("\nInitialization time: {:.2f} s \n".format(end_initialization - start_initialization))
    logger_func('Initialization ends', 'i')
    logger_func('Initialization time: '+str((end_initialization - start_initialization)), 'i')

    if output_settings.restart == True and thermostat == False:
        init_steps = 0
    else:
        init_steps = md_variables.init_steps
        
    print('Number of initialization steps:', init_steps,'\n')
    logger_func('No. of  initialization steps '+str(init_steps), 'i')

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
                logger_func('End of thermostatting', 'i')
                thermostat = False
                counter = counter + 1
            
            if output_settings.print_solute:
                df = pd.DataFrame(grid.particles.pos, columns=['x', 'y', 'z'])
                df['vx'] = grid.particles.vel[:, 0]
                df['vy'] = grid.particles.vel[:, 1]
                df['vz'] = grid.particles.vel[:, 2]
                df['fx_elec'] = grid.particles.forces[:, 0]
                df['fy_elec'] = grid.particles.forces[:, 1]
                df['fz_elec'] = grid.particles.forces[:, 2]
                df['charge'] = grid.particles.charges
                df['iter'] = i - init_steps
                df['particle'] = np.arange(N_p)
                df.to_csv(
                    ofiles.file_output_solute, index=False, header=False, mode='a',
                    columns=['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec']
                    )
                                    
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

    if output_settings.print_solute:
        ofiles.file_output_solute.close()

    if output_settings.print_energy:
        ofiles.file_output_energy.close()

    if output_settings.print_temperature:
        ofiles.file_output_temperature.close()

    if output_settings.generate_restart_file:
        restart_file = generate_restart(grid_setting, output_settings, iter_restart)
        print('Restart file generated: ', restart_file)
        logger_func('Restart file generated: '+str(restart_file), 'i')
        
    end_time = time.time()
    print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))
    logger_func('Total time taken: '+str((end_time - begin_time)), 'i')
    logger_func('--------------END RUN---------------------', 'i')
    logger_func('-----------------------------------', 'i')
