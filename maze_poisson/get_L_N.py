def calculate_length_and_grid_points(Np, ref_Np=250, ref_L=20.9, ref_N=100, m_Na=22.99, m_Cl=35.453):
    """
    Calculate the side length (L) and the number of grid points (N) for a given number of particles (Np)
    based on the density of a reference system with half Na and half Cl atoms.

    Parameters:
    Np (int): Number of particles.
    ref_Np (int): Number of particles in the reference system (default is 250).
    ref_L (float): Side length of the reference system in Angstroms (default is 20.9).
    ref_N (int): Number of grid points per side in the reference system (default is 100).
    m_Na (float): Mass of a sodium atom (default is 22.99).
    m_Cl (float): Mass of a chlorine atom (default is 35.453).

    Returns:
    (float, int, float): Tuple containing the side length (L) in Angstroms, the number of grid points (N), and the density.
    """
    # Calculate the average mass of a particle in the reference system
    avg_mass_ref = (m_Na + m_Cl) / 2
    
    # Calculate the reference density in terms of mass per volume
    ref_density = (ref_Np * avg_mass_ref) / (ref_L ** 3)

    # Calculate the average mass of a particle in the new system (which is the same as in the reference system)
    avg_mass_new = avg_mass_ref

    # Calculate the side length for the given number of particles
    L = ((Np * avg_mass_new) / ref_density) ** (1/3)

    # Calculate the number of grid points, keeping the same density of grid points per unit length as the reference
    N = int(round(L / ref_L * ref_N))
    
    # Calculate the new density for verification
    new_density = (Np * avg_mass_new) / (L ** 3)

    return L, N, new_density

m_Na=22.99
m_Cl=35.453
# Example usage
Np_values = [2, 16, 54, 128, 250, 432, 686, 1024]
results = {Np: calculate_length_and_grid_points(Np) for Np in Np_values}
avg_mass_ref = (m_Na + m_Cl) / 2
for Np, (L, N, density) in results.items():
    print(f"Np = {Np}: L = {L:.2f} Å, N = {N} grid points, Density = {density:.5f} g/Å^3", Np / L**3)
    #print(L / N)


