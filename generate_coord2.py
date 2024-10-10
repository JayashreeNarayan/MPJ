import numpy as np
filename = 'input_files/input_coord1000.csv'

# Define the dimensions of the box
L = 46.04#19.92#21.02#16.72#6.64#13.27#5.6 * 2

# Number of particles
num_particles = 1000

# Calculate the number of particles per dimension
num_per_dim = 10

# Calculate the spacing between particles
spacing = L / num_per_dim
#print(spacing)
# Generate particle coordinates
# Shuffle coordinates to avoid any pattern
#random.shuffle(coordinates)

# Header for the CSV file
header = "charge,mass,radius,x,y,z\n"
h = 1e-6
indices_x = np.arange(spacing/2,L+h,spacing)
indices_y = np.arange(spacing/2,L+h,spacing) 
indices_z = np.arange(spacing/2,L+h,spacing) 

m_Na = 22.99
m_Cl = 35.453
#indices_z = np.arange(spacing,L+h,spacing * 2)

# Write coordinates to CSV file
with open(filename, "w") as f:
    f.write(header)
    idx = 0
    sgn_y = 0
    sgn_x = 0
    for x in indices_x:
        sgn_x +=1
        for y in indices_y:
            sgn_y += 1
            for z in indices_z:
                charge = (-1) ** (idx + sgn_x + sgn_y) # Alternate the charge between 1 and -1
                if charge > 0:
                    mass = m_Na
                else:
                    mass = m_Cl
                radius = 1
                f.write(f"{charge},{mass},{radius},{x},{y},{z}\n")
                idx += 1
                

print("CSV file '" + str(filename) + ".csv' has been generated.")
