from ..input import grid_setting, md_variables

N = grid_setting.N
N_p = grid_setting.N_p
dt = md_variables.dt
init = md_variables.initialization
preconditioning = md_variables.preconditioning
omega = '1'
### Output file path ###
#'_dt'+str(dt) + '
#

if init == 'CG':
    if preconditioning == 'Yes':
        #output_field = 'data/data_init_CG/preconditioned_lastresult/particles4_L25/output_N' + str(N) + '.csv'
        #output_field = 'data/test_plot/output_N' + str(N) + '.csv'
        output_field = 'data/omega_analysis/omega_'+ omega + '/output_N' + str(N) + '.csv'
    else:
        output_field = 'data/data_init_CG_random/no_preconditioned/particles4_L25/output_N' + str(N) + '.csv'
else:
    output_field = 'data/test_plot/output_N' + str(N) + '.csv'
    #output_field = 'data/test_neigh_afterholiday/output_N' + str(N) + '.csv'
    #output_field = 'data/output_N' + str(N) + '.csv'

file_output_field = open(output_field, 'r+')
file_output_field.write("N,iter,x,MaZe\n")


if init == 'CG':
    if preconditioning == 'Yes':
        #output_time = 'data/data_init_CG/preconditioned_lastresult/particles4_L25/time_N' + str(N) + '.csv'
       #output_time = 'data/test_plot/time_N' + str(N) + '.csv'
        output_time = 'data/omega_analysis/omega_'+ omega + '/time_N' + str(N) + '.csv'
    else:
        output_time = 'data/data_init_CG/no_preconditioned/particles4_L25/time_N' + str(N) + '.csv'
else:
    output_time = 'data/test_plot/time_N' + str(N) + '.csv'
    #output_time = 'data/test_neigh_afterholiday/time_N' + str(N) + '.csv'
    #output_time = 'data/time_N' + str(N) + '.csv'

file_output_time = open(output_time, 'r+')
file_output_time.write("iter,Verlet,SetCharges\n") #,InitializeMatrix\n")



if init == 'CG':
    if preconditioning == 'Yes':
        #output_convergence = 'data/data_init_CG/preconditioned_lastresult/particles4_L25/convergence_N' + str(N) + '.csv'
        #output_convergence = 'data/test_plot/convergence_N' + str(N) + '.csv'
        output_convergence = 'data/omega_analysis/omega_'+ omega + '/convergence_N' + str(N) + '.csv'
    else:
        output_convergence = 'data/data_init_CG/no_preconditioned/particles4_L25/convergence_N' + str(N) + '.csv'
else:
    output_convergence = 'data/test_plot/convergence_N' + str(N) + '.csv'
    #output_time = 'data/test_neigh_afterholiday/time_N' + str(N) + '.csv'
    #output_time = 'data/time_N' + str(N) + '.csv'

file_output_convergence = open(output_convergence, 'r+')
file_output_convergence.write("iter_convergence,accuracy\n")


if preconditioning == 'Yes':
    output_solute = 'data/omega_analysis/omega_'+ omega + '/output_solute_N' + str(N) + '.csv'
else:
    output_solute = 'data/not_preconditioned/output_solute_N' + str(N) + '.csv'

file_output_solute = open(output_solute, 'r+')
file_output_solute.write("charge,iter,particle,x,y,z,vx,vy,vz,fx,fy,fz\n")


