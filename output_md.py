from input import grid_setting, output_settings

N = grid_setting.N
path = output_settings.path

if output_settings.print_field:
    output_field = path + 'field_N' + str(N) + '.csv'
    file_output_field = open(output_field, 'r+')
    file_output_field.write("iter,x,MaZe\n")

if output_settings.print_performance:
    output_performance = path + 'performance_N' + str(N) + '.csv'
    file_output_performance = open(output_performance, 'r+')
    file_output_performance.write("iter,time,n_iters\n") 

if output_settings.print_energy:
    output_energy = path + 'energy_N' + str(N) + '.csv'
    file_output_energy = open(output_energy, 'r+')
    file_output_energy.write("t,E,K,V\n")

if output_settings.print_temperature:
    output_temperature = path + 'temperature_N' + str(N) + '.csv'
    file_output_temperature = open(output_temperature, 'r+')
    file_output_temperature.write("t,T\n")

if output_settings.print_solute:
    output_solute = path + 'solute_N' + str(N) + '.csv'
    file_output_solute = open(output_solute, 'r+')
    file_output_solute.write("charge,iter,particle,x,y,z,vx,vy,vz\n") #,fx,fy,fz\n")

    
