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
@Np_option
def time_vs_N3(np):
    plot_time_iterNgrid(np)

@plot.command()
@Np_option
def iter_vs_N3(np):
    plot_convNgrid(np)
    
@plot.command()
def time_vs_Np():
    plot_scaling_particles_time_iters()

@plot.command()
def iter_vs_Np():
    plot_scaling_particles_conv()

@plot.command()
@filename_argument
def visualize(filename):
    visualize_particles(filename)

@plot.command()
# SANITY CHECK GRAPH
def iter_vs_thread():
    iter_vs_threads()

@plot.command()
def time_vs_thread():
    time_vs_threads()

@plot.command()
def iters_vs_tol():
    iter_vs_tol()