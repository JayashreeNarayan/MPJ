import csv
import numpy as np

a0 = 0.529177210903 #Angstrom
n = 64 

def convert_csv_to_xyz(input_csv_file, output_xyz_file):
    # Dictionary to map charges to element symbols
    charge_to_element = {1: 'Na', -1: 'Cl'}
    frame = 0
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_xyz_file, 'w') as xyz_file:
            num_particles = 0
            for row in csv_reader:
                charge = int(row['charge'])
                element = charge_to_element.get(charge, 'X')  # Default to 'X' if charge is neither 1 nor -1
                x, y, z = float(row['x']),float(row['y']),float(row['z']) 
                if num_particles % n == 0:
                    xyz_file.write(str(n) + '\n')
                    xyz_file.write(str(frame) + '\n')
                    frame = frame + 1
                line = f"{element} {np.round(x * a0, decimals=3)} {np.round(y * a0, decimals=3)} {np.round(z * a0, decimals=3)}\n"
                xyz_file.write(line)
                num_particles += 1

            print(f"Conversion completed. {num_particles / 64} steps written to {output_xyz_file}")

# Example usage:
input_csv_file = '../data/test_github/solute_N50.csv'
output_xyz_file = '../data/test_github/data_N50.xyz'
convert_csv_to_xyz(input_csv_file, output_xyz_file)
