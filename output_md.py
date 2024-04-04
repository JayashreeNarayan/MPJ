from input import grid_setting, md_variables

N = grid_setting.N
dt = md_variables.dt
init = md_variables.initialization
preconditioning = md_variables.preconditioning
### Output file path ###
#'_dt'+str(dt) + '
#

#path = 'data_md/test/test_8_T1550/'
#path = 'data_md/test/test_64_T1550/test_10K/'
#path = 'data_md/test/test_64_T1550/dt_1/20_30K/'
#path = 'new_data/test_autocorrvel/dt_10/'
path = 'new_data/test_github/'
if init == 'CG':
    if preconditioning == 'Yes':
        output_field = path + 'output_N' + str(N) + '.csv'


file_output_field = open(output_field, 'r+')
file_output_field.write("iter,x,MaZe\n")


if init == 'CG':
    if preconditioning == 'Yes':
        #output_time = 'data/data_init_CG/preconditioned_lastresult/particles4_L25/time_N' + str(N) + '.csv'
       #output_time = 'data/test_plot/time_N' + str(N) + '.csv'
        output_time = path + 'time_N' + str(N) + '.csv'
    else:
        output_time = 'data/data_init_CG/no_preconditioned/particles4_L25/time_N' + str(N) + '.csv'
else:
    output_time = 'data/test_plot/time_N' + str(N) + '.csv'
    #output_time = 'data/test_neigh_afterholiday/time_N' + str(N) + '.csv'
    #output_time = 'data/time_N' + str(N) + '.csv'

file_output_time = open(output_time, 'r+')
file_output_time.write("iter,Verlet\n") #,InitializeMatrix\n")



if init == 'CG':
    if preconditioning == 'Yes':
        #output_convergence = 'data/data_init_CG/preconditioned_lastresult/particles4_L25/convergence_N' + str(N) + '.csv'
        #output_convergence = 'data/test_plot/convergence_N' + str(N) + '.csv'
        output_convergence = path + 'convergence_N' + str(N) + '.csv'
    else:
        output_convergence = 'data/data_init_CG/no_preconditioned/particles4_L25/convergence_N' + str(N) + '.csv'
else:
    output_convergence = 'data/test_plot/convergence_N' + str(N) + '.csv'
    #output_time = 'data/test_neigh_afterholiday/time_N' + str(N) + '.csv'
    #output_time = 'data/time_N' + str(N) + '.csv'

file_output_convergence = open(output_convergence, 'r+')
file_output_convergence.write("iter_convergence,accuracy\n")



if preconditioning == 'Yes':
    output_solute = path + 'output_solute_N' + str(N) + '.csv'
else:
    output_solute = 'data/not_preconditioned/output_solute_N' + str(N) + '.csv'

file_output_solute = open(output_solute, 'r+')
file_output_solute.write("charge,iter,particle,x,y,z,vx,vy,vz\n") #,fx,fy,fz\n")


file_output_energy = open(path + 'energy_N' + str(N) + '.csv', 'r+')
file_output_energy.write("t,E,K,V\n")

#file_output_xyz = open(path + 'xyz_N' + str(N) + '.csv', 'r+')
#file_output_xyz.write(grid_setting.N_p)
#file_output_xyz.write('\n')
