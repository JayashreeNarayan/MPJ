import numpy as np
filename = 'input_coord250.csv'

# Define the dimensions of the box
L = 20.9#6.64#13.27#5.6 * 2

# Calculate the number of particles per dimension
num_per_dim = 5

# Calculate the spacing between particles
spacing = L / (num_per_dim)
spacing_10 = L / (2 * num_per_dim)
#print(spacing)
# Generate particle coordinates
# Shuffle coordinates to avoid any pattern
#random.shuffle(coordinates)

# Header for the CSV file
header = "charge,mass,radius,x,y,z\n"
h = 1e-6
indices_x = np.sort(np.concatenate((np.arange(spacing/2,L+h,spacing),np.arange(spacing/8,L+h,spacing))))
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
    for i,x in enumerate(indices_x):
        sgn_x += 1
        for j,y in enumerate(indices_y):
            sgn_y += 1
            for k,z in enumerate(indices_z):
                idx += 1
                overall_idx = i * num_per_dim * num_per_dim + j * num_per_dim + k
                #charge = (-1) ** (idx + sgn_x % 2 + sgn_y) # Alternate the charge between 1 and -1
                charge = (-1)**overall_idx
                if charge > 0:
                    mass = m_Na
                else:
                    mass = m_Cl
                radius = 1
                f.write(f"{charge},{mass},{radius},{x},{y},{z}\n")
                
                
                

print("CSV file '" + str(filename) + ".csv' has been generated.")
