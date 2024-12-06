import click

from .maze import maze
from .plotters.plot_force import plot_force, plot_forcemod
from .plotters.plot_T_E_tot import *
from .plotters.plot_scaling import *

filename_argument = click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='CSV_FILE',
    )

dt_option = click.option(
    '--dt', 'dt',
    type=float,
    default=0.25,
    help='Timestep for the solute evolution given in fs and converted in a.u.',
    )

therm_option = click.option(
    '--therm', 'therm',
    type=str,
    default='N',
    help='Did you previously run with a Thermostat? (Y/N)',
    )

Np_option = click.option(
    '--np', 'np',
    type=int,
    default=216,
    help='Value of total number of particles (N_p)',
    )

@maze.group()
def plot():
    pass

@plot.command()
@filename_argument
@dt_option
@therm_option
def force(filename, dt, therm): 
    plot_force(filename, dt, therm)

@plot.command()
@filename_argument
@dt_option
@therm_option
def forcemod(filename, dt, therm):
    plot_forcemod(filename, dt, therm)
    
@plot.command()
@filename_argument
@dt_option
@therm_option
def temperature(filename, dt, therm):
    PlotT(filename, dt, therm)
    
@plot.command()
@filename_argument
@dt_option
@therm_option
def energy(filename, dt, therm):
    plot_Etot_trp(filename, dt, therm)

@plot.command()
@therm_option
@Np_option
def time_vs_N3(np, therm):
    plot_time_iterNgrid(np, therm)

@plot.command()
@therm_option
@Np_option
def n_iter_vs_N3(np, therm):
    plot_convNgrid(np, therm)
    
@plot.command()
@filename_argument
@therm_option
def time_vs_Np(filename, therm):
    plot_scaling_particles_time_iters(filename, therm)

@plot.command()
@filename_argument
def visualize(filename):
    visualize_particles(filename)

