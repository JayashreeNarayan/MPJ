import os

class OutputFiles:
    file_output_field = None
    file_output_performance = None
    file_output_iters = None
    file_output_energy = None
    file_output_temperature = None
    file_output_solute = None
    file_output_tot_force = None

def generate_output_files(grid):
    N = grid.N
    output_settings = grid.output_settings
    path = output_settings.path

    output_files = OutputFiles()

    if output_settings.print_field:
        output_field = path + 'field_N' + str(N) + '.csv'
        if os.path.exists(output_field):
            open(output_field, 'w').close()  # Clear file content if it exists
        else:
            open(output_field, 'w').close()  # Create the file
        output_files.file_output_field = open(output_field, 'r+')
        output_files.file_output_field.write("iter,x,MaZe\n")

    if output_settings.print_performance:
        output_performance = path + 'performance_N' + str(N) + '.csv'
        if os.path.exists(output_performance):
            open(output_performance, 'w').close()  # Clear file content if it exists
        else:
            open(output_performance, 'w').close()  # Create the file
        output_files.file_output_performance = open(output_performance, 'r+')
        output_files.file_output_performance.write("iter,time,n_iters\n")


    if output_settings.print_iters:
        output_iters = path + 'iters_N' + str(N) + '.csv'
        if os.path.exists(output_iters):
            open(output_iters, 'w').close()  # Clear file content if it exists
        else:
            open(output_iters, 'w').close()  # Create the file
        output_files.file_output_iters = open(output_iters, 'r+')
        output_files.file_output_iters.write("iter,max_sigma,norm\n")


    if output_settings.print_energy:
        output_energy = path + 'energy_N' + str(N) + '.csv'
        if os.path.exists(output_energy):
            open(output_energy, 'w').close()  # Clear file content if it exists
        else:
            open(output_energy, 'w').close()  # Create the file
        output_files.file_output_energy = open(output_energy, 'r+')
        #file_output_energy.write("iter,E,K,V_elec,V_notelec\n")
        output_files.file_output_energy.write("iter,K,V_notelec\n")

    if output_settings.print_temperature:
        output_temperature = path + 'temperature_N' + str(N) + '.csv'
        if os.path.exists(output_temperature):
            open(output_temperature, 'w').close()  # Clear file content if it exists
        else:
            open(output_temperature, 'w').close()  # Create the file
        output_files.file_output_temperature = open(output_temperature, 'r+')
        output_files.file_output_temperature.write("iter,T\n")

    if output_settings.print_solute:
        output_solute = path + 'solute_N' + str(N) + '.csv'
        if os.path.exists(output_solute):
            open(output_solute, 'w').close()  # Clear file content if it exists
        else:
            open(output_solute, 'w').close()  # Create the file
        output_files.file_output_solute = open(output_solute, 'r+')
        output_files.file_output_solute.write("charge,iter,particle,x,y,z,vx,vy,vz,fx,fy,fz\n")

    if output_settings.print_tot_force:
        output_tot_force = path + 'tot_force_N' + str(N) + '.csv'
        if os.path.exists(output_tot_force):
            open(output_tot_force, 'w').close()  # Clear file content if it exists
        else:
            open(output_tot_force, 'w').close()  # Create the file
        output_files.file_output_tot_force = open(output_tot_force, 'r+')
        output_files.file_output_tot_force.write("iter,Fx,Fy,Fz\n")

    return output_files
