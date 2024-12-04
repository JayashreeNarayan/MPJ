import numpy as np

# Set parameters for NaCl simulation
# nmol = 250  # Total number of particles (125 NaCl molecules)
# ndim = 3    # Number of dimensions
# boxl = 19.659 # Length of the simulation box in angstroms
# partner = np.arange(nmol)  # Example partner array
# filen = 'input_files_new/input_coord' + str(nmol)  # Output filename

# natoms = 2  # Number of atom types (Na and Cl)

# # Run the simulation
# lattice(nmol, ndim, boxl, partner, filen, natoms)

def lattice(nmol, ndim, boxl, natoms, partner):
    na = int((nmol / 2) ** (1/3)) + 1
    del_spacing = 1.0 / na
    bond = 0.2 / boxl
    # print(f'LATTICE : {na}, {del_spacing}, {bond}')

    r = np.zeros((ndim, nmol))
    itel = 0

    # Define charges, masses, and radii for Na and Cl atoms
    charge_Na, mass_Na, radius_Na = 1, 22.99, 1.0  # Example values
    charge_Cl, mass_Cl, radius_Cl = -1, 35.453, 1.0  # Example values

    # Define arrays for charges, masses, and radii
    charges = np.zeros(nmol)
    masses = np.zeros(nmol)
    radii = np.zeros(nmol)

    # Place particles A on a BCC lattice
    for i in range(na):
        for j in range(na):
            for k in range(na):
                if itel < nmol - 1:
                    r[0, itel] = k * del_spacing
                    r[1, itel] = j * del_spacing
                    r[2, itel] = i * del_spacing
                    charges[itel], masses[itel], radii[itel] = charge_Na, mass_Na, radius_Na
                    
                    r[0, itel + 1] = (k + 0.5) * del_spacing
                    r[1, itel + 1] = (j + 0.5) * del_spacing
                    r[2, itel + 1] = (i + 0.5) * del_spacing
                    charges[itel + 1], masses[itel + 1], radii[itel + 1] = charge_Cl, mass_Cl, radius_Cl
                    
                    itel += 2
    
    # If there are leftover particles
    if itel < nmol:
        raise ValueError('lattice.bcc: bad number of particles')
        # for i in range(na):
        #     for j in range(na):
        #         for k in range(na):
        #             if itel < nmol:
        #                 itel += 1
        #                 r[0, itel] = i * del_spacing + 0.25 * del_spacing
        #                 r[1, itel] = j * del_spacing + 0.25 * del_spacing
        #                 r[2, itel] = k * del_spacing + 0.25 * del_spacing
        #                 charges[itel], masses[itel], radii[itel] = charge_Na, mass_Na, radius_Na

    # If there are 2 atom types, place B particles close to A particles
    if natoms == 2:
        for i in range(nmol):
            for k in range(ndim):
                ronf = np.random.random()
                r[k, partner[i]] = r[k, i] + bond * (ronf - 0.5)

    # Save particle positions to a file with the desired structure
    # filename = f'{filen}.csv'
    # with open(filename, 'w') as f:
    # file.write("charge,mass,radius,x,y,z\n")  # Write the header
    for i in range(nmol):
        for l in range(ndim):
            r[l, i] = r[l, i] - np.rint(r[l, i])
                # Keep particles within the unit box

            # Check that positions are within the -0.5 to 0.5 range
            if r[l, i] > 0.5 or r[l, i] < -0.5:
                print(f'Problem with the initial conf: not in -0.5:0.5, {i}, {l}')
                raise ValueError('Position out of range -0.5:0.5')

        # Convert to box units and center around [0, L] by adding boxl/2
        x, y, z = r[0, i] * boxl + boxl/2, r[1, i] * boxl + boxl/2, r[2, i] * boxl + boxl/2

        # Write charge, mass, radius, x, y, z to the file
        # file.write(f"{charges[i]},{masses[i]},{radii[i]},{x},{y},{z}\n")

    return x,y,z, charges, masses, radii

