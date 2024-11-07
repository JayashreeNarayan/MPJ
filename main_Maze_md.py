##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

import numpy as np
from grid import *
from input import md_variables, grid_setting, a0, output_settings
from indices import dict_indices_CoordTon
from verlet import VerletPoisson, VerletPoissonBerendsen, PrecondLinearConjGradPoisson, VerletSolutePart1, VerletSolutePart2, OVRVO_part1, OVRVO_part2
import time
from tqdm import tqdm
from restart import generate_restart

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

if output_settings.print_tot_force:
    from output_md import file_output_tot_force


# get variables from input
h = grid_setting.h
L = grid_setting.L
N = grid_setting.N
N_steps = md_variables.N_steps
stride = md_variables.stride
initialization = md_variables.initialization
omega = md_variables.omega
thermostat = md_variables.thermostat
dt = md_variables.dt
preconditioning = md_variables.preconditioning
elec = md_variables.elec
V = 27.211386245988
tol = md_variables.tol

# initialize grid by inserting particles in the system
grid = Grid()

# print relevant info
print('\nSimulation with N =', N, 'with N_steps =', N_steps, 'and tol =', md_variables.tol)
#for i, particle in enumerate(grid.particles):
#    print('Position particle', i + 1, '=', particle.pos)
#    print('Velocity particle', i + 1, '=', particle.vel)
print('\nInitialization is done with CG and preconditioning:', preconditioning)
print('\nParameters:\nh =', h, "\ndt =", dt, "\nstride =", stride, '\nL =', L, '\nomega =', omega, '\ngamma =', md_variables.gamma)
print('\nPotential:', md_variables.potential)
print('\nElec:', elec, '\tNotElec: ', not_elec,'\n')
print('\nTemperature:',T,' K\tNumerical density:',N_p / L**3,' a.u.')
print('\nPrint solute:', output_settings.print_solute, '\tPrint field: ', output_settings.print_field, '\tPrint tot_force:', output_settings.print_tot_force,
      '\nPrint energy:', output_settings.print_energy, '\tPrint temperature:', output_settings.print_temperature,
      '\tPrint performance:', output_settings.print_performance,'\tRestart:', output_settings.restart,'\n')
print('\nThermostat:', thermostat)

################################ STEP 0 Verlet ##########################################
#########################################################################################

q_tot = 0
#compute 8 nearest neighbors for any particle
for particle in grid.particles:
    particle.NearestNeigh()
    q_tot = q_tot + particle.charge

print('Total charge q = ',q_tot)

# set charges with the weight function
grid.SetCharges()

# initialize the electrostatic field with CG                  
if preconditioning == "Yes":
    grid.phi_prev, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol)

# AGGIUSTA
#grid.LinkedCellInit(grid.particles[0].r_cutoff)

if not_elec:
    grid.ComputeForceNotElecBasic()
    #grid.ComputeForceNotElecLC()

for particle in grid.particles:
    if elec:
        particle.ComputeForce_FD(grid, prev=True) 


################################ STEP 1 Verlet ##########################################
#########################################################################################

# Velocity Verlet for the solute
if md_variables.integrator == 'OVRVO':
    grid.particles = OVRVO_part1(grid.particles, thermostat = thermostat)
else:
    grid.particles = VerletSolutePart1(grid.particles, thermostat=thermostat)

# compute 8 nearest neighbors for any particle
for p, particle in enumerate(grid.particles):
        particle.NearestNeigh()

# set charges with the weight function
grid.SetCharges()
    
if preconditioning == "Yes":
    grid.phi, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol, x0=grid.phi_prev)

if md_variables.integrator == 'OVRVO':
    grid.particles = OVRVO_part2(grid, thermostat = thermostat)
else:
    grid = VerletSolutePart2(grid)


################################ FINE INIZIALIZZAZIONE ##########################################
#########################################################################################

X = np.arange(0, L, h)
j = int(grid.particles[0].pos[1] / h)
k = int(grid.particles[0].pos[2] / h)

end_initialization = time.time()
print("\nInitialization time: {:.2f} s \n".format(end_initialization - start_initialization))

if output_settings.restart == True and thermostat == False:
    init_steps = 0
else:
    init_steps = md_variables.init_steps
    
print('Number of initialization steps:', init_steps,'\n')

y = np.zeros(N_tot) 

######################################### Verlet ############################################
#############################################################################################

counter = 0 
old_pos = []

# iterate over the number of steps (i.e times I move the particle 1)
for i in tqdm(range(N_steps)):
    #print('\nStep = ', i, ' t elapsed from init =', time.time() - end_initialization)
    for particle in grid.particles:
        old_pos.append(particle.pos)

    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part1(grid.particles, thermostat = thermostat)
    else:
        grid.particles = VerletSolutePart1(grid.particles, thermostat = thermostat)

    if elec:
        # compute 8 nearest neighbors for any particle
        for particle in grid.particles:
            particle.NearestNeigh()
    
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
      
        for j, particle in enumerate(grid.particles):
            tot_force = tot_force + particle.force + particle.force_notelec
            
            if np.linalg.norm(particle.force_notelec) > 0.:
                print('\nStep = ', i, ' t elapsed from init =', time.time() - end_initialization)
                print(particle.force, particle.force_notelec)
        
        file_output_tot_force.write(str(i) + ',' + str(tot_force[0]) + ',' + str(tot_force[1]) + ','+ str(tot_force[2]) + "\n") 

    grid.Energy(print_energy=output_settings.print_energy, iter=i)
    grid.Temperature(print_temperature=output_settings.print_temperature, iter=i)
    
            
    if i % stride == 0 and i >= init_steps:
        if counter == 0 and thermostat == True:
            print('End of thermostatting')
            thermostat = False
            counter = counter + 1
        
        if output_settings.print_solute:
            for p in range(grid.N_p):
                file_output_solute.write(str(grid.particles[p].charge) + ',' + str(i - init_steps) + ',' + str(p) 
                                + ',' + str(grid.particles[p].pos[0]) + ',' + str(grid.particles[p].pos[1]) + ',' + str(grid.particles[p].pos[2]) # pos are printed in a.u., then converted to angs in the convert_to_xyz file
                                + ','  + str(grid.particles[p].vel[0]) + ',' + str(grid.particles[p].vel[1]) + ',' + str(grid.particles[p].vel[2])  #vel are in a.u.
                                + ','  + str(grid.particles[p].force[0]) + ',' + str(grid.particles[p].force[1]) + ',' + str(grid.particles[p].force[2]) + '\n')
                                #+ ','  + str(grid.particles[p].force_notelec[0]) + ',' + str(grid.particles[p].force_notelec[1]) + ',' + str(grid.particles[p].force_notelec[2]) + '\n')
                                #+ ','  + str(grid.particles[p].force[0] + grid.particles[p].force_notelec[0]) + ',' + str(grid.particles[p].force[1]+ grid.particles[p].force_notelec[1]) + ',' + str(grid.particles[p].force[2]+ grid.particles[p].force_notelec[2]) + '\n')          
        
        if output_settings.print_performance and elec:
            file_output_performance.write(str(i - init_steps) + ',' + str(end_Verlet - start_Verlet) + ',' + str(iter_conv) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n")
                    
        if output_settings.print_field and elec:
            field_x_MaZe = np.array([grid.phi[dict_indices_CoordTon[tuple([l, j, k])]] for l in range(N)])
            for n in range(N):
                file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')

# close output files
if output_settings.print_field:
    file_output_field.close()

if output_settings.print_performance:
    file_output_performance.close()

#if output_settings.print_iters:
#    file_output_iters.close()

if output_settings.print_solute:
    file_output_solute.close()

if output_settings.print_energy:
    file_output_energy.close()

if output_settings.print_temperature:
    file_output_temperature.close()

if output_settings.generate_restart_file:
    restart_file = generate_restart()
    print('Restart file generated: ', restart_file)

      
end_time = time.time()
print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))

