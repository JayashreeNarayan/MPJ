import sys

import click
import numpy as np
import pandas as pd

from .maze import maze
from .utilities.bcc_generate import generate_bcc_positions as bcc_positions
from .utilities.convert_to_xyz import convert_csv_to_xyz
from .utilities.generate_bcc import lattice as bcc_lattice
from .utilities.get_L_N import get_L_N as _get_L_N


@maze.group()
def utils():
    pass

lattice_choices = ['bcc']
lattice_option = click.option(
    '--lattice', '-l',
    'lattice_type',
    type=click.Choice(lattice_choices),
    # required=True,
    default='bcc',
    help=f'Type of lattice to generate. Supported options: {", ".join(lattice_choices)}',
)
output_option = click.option(
    '--output', '-o', 'output',
    default=None,
    type=click.Path(),
    help='Output file for the generated lattice.',
)
input_option = click.option(
    '--input', '-i', 'input',
    default=None,
    type=click.Path(),
    help='Input file to convert.',
)

@utils.command()
@lattice_option
@output_option
@click.argument('nmol', type=int)
@click.argument('ndim', type=int)
@click.argument('boxl', type=float)
@click.argument('natoms', type=int)
def generate_lattice(nmol, ndim, boxl, natoms, lattice_type, output):
    if output is None:
        output = sys.stdout

    partner = np.arange(nmol)

    if lattice_type == 'bcc':
        x,y,z, charges, masses, radii = bcc_lattice(nmol, ndim, boxl, natoms, partner=partner)
    else:
        raise ValueError(f'Unsupported lattice type: {lattice_type}')

    pd.DataFrame({
        'charge': charges,
        'mass': masses,
        'radius': radii,
        'x': x,
        'y': y,
        'z': z,
    }).to_csv(output, index=False)


@utils.command()
@lattice_option
@output_option
@click.argument('box_size', type=float)
@click.argument('num_particles', type=int)
@click.argument('epsilon', type=float, default=0.2)
def generate_positions(box_size, num_particles, epsilon, lattice_type, output):  # main function to use
    if output is None:
        output = sys.stdout
    if lattice_type == 'bcc':
        positions = bcc_positions(box_size, num_particles, epsilon)
    else:
        raise ValueError(f'Unsupported lattice type: {lattice_type}')
        
    m_Na = 22.99
    m_Cl = 35.453

    pd.DataFrame({
        'charge': np.where(np.arange(num_particles) % 2 == 0, 1, -1),
        'mass': np.where(np.arange(num_particles) % 2 == 0, m_Na, m_Cl),
        'radius': 1,
        'x': positions[:, 0],
        'y': positions[:, 1],
        'z': positions[:, 2],
    }).to_csv(output, index=False)

@utils.command()
@output_option
@input_option
def convert(input, output):
    # Dictionary to map charges to element symbols
    if input is None:
        input = sys.stdin.fileno()
    if output is None:
        output = sys.stdout.fileno()
    convert_csv_to_xyz(input, output)

@utils.command()
def get_L_N():
    _get_L_N()
