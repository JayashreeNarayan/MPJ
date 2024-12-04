import atexit
import os


class OutputFiles:
    file_output_field = None
    file_output_performance = None
    file_output_iters = None
    file_output_energy = None
    file_output_temperature = None
    file_output_solute = None
    file_output_tot_force = None

open_files = []

def generate_output_file(out_path, overwrite=True):
    if os.path.exists(out_path):
        if overwrite:
            os.remove(out_path)
        else:
            raise ValueError(f"File {out_path} already exists")
    res = open(out_path, 'w')
    open_files.append(res)
    return res


def generate_output_files(grid):
    N = grid.N
    output_settings = grid.output_settings
    md_variables = grid.md_variables
    path = output_settings.path

    output_files = OutputFiles()

    if md_variables.thermostat:
        thermostat_path = os.path.join(path, 'Thermostatted')
        os.makedirs(thermostat_path, exist_ok=True)
        path = thermostat_path  # if thermostating is true then this has to take place.

    if output_settings.print_field:
        field_path = os.path.join(path, 'Field/')
        os.makedirs(field_path, exist_ok=True)
        output_field = os.path.join(field_path, 'field_N' + str(N) + '.csv')
        output_files.file_output_field = generate_output_file(output_field)
        output_files.file_output_field.write("iter,x,MaZe\n")

    if output_settings.print_performance:
        performance_path = os.path.join(path, 'Performance/')
        os.makedirs(performance_path, exist_ok=True)
        output_performance = os.path.join(performance_path, 'performance_N' + str(N) + '.csv')
        output_files.file_output_performance = generate_output_file(output_performance)
        output_files.file_output_performance.write("iter,time,n_iters\n")

    if output_settings.print_iters:
        iters_path = os.path.join(path, 'Iterations/')
        os.makedirs(iters_path, exist_ok=True)
        output_iters = os.path.join(iters_path, 'iters_N' + str(N) + '.csv')
        output_files.file_output_iters = generate_output_file(output_iters)
        output_files.file_output_iters.write("iter,max_sigma,norm\n")

    # prints only kinetic and non electrostatic potential
    if output_settings.print_energy:
        energy_path = os.path.join(path, 'Energy/')
        os.makedirs(energy_path, exist_ok=True)
        output_energy = os.path.join(energy_path, 'energy_N' + str(N) + '.csv')
        output_files.file_output_energy = generate_output_file(output_energy)
        output_files.file_output_energy.write("iter,K,V_notelec\n")

    if output_settings.print_temperature:
        temperature_path = os.path.join(path, 'Temperature/')
        os.makedirs(temperature_path, exist_ok=True)
        output_temperature = os.path.join(temperature_path, 'temperature_N' + str(N) + '.csv')
        output_files.file_output_temperature = generate_output_file(output_temperature)
        output_files.file_output_temperature.write("iter,T\n")

    if output_settings.print_solute:
        solute_path = os.path.join(path, 'Solute/')
        os.makedirs(solute_path, exist_ok=True)
        output_solute = os.path.join(solute_path, 'solute_N' + str(N) + '.csv')
        output_files.file_output_solute = generate_output_file(output_solute)
        output_files.file_output_solute.write("charge,iter,particle,x,y,z,vx,vy,vz,fx_elec,fy_elec,fz_elec\n")

    if output_settings.print_tot_force:
        force_path = os.path.join(path, 'Tot_Force/')
        os.makedirs(force_path, exist_ok=True)
        output_tot_force = os.path.join(force_path, 'tot_force_N' + str(N) + '.csv')
        output_files.file_output_tot_force = generate_output_file(output_tot_force)
        output_files.file_output_tot_force.write("iter,Fx,Fy,Fz\n")

    return output_files

@atexit.register
def close_output_files():
    not_closed = []
    while open_files:
        app = open_files.pop()
        try:
            app.close()
        except Exception as e:
            print(f"Error closing file: {e}")
            not_closed.append(app)

    if not_closed:
        open_files.extend(not_closed)
        print("Some files could not be closed")
            