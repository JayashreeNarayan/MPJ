import click

from .maze import maze
from .plotters.plot_force import plot_force, plot_forcemod
from .plotters.plot_T_E_tot import PlotT, store_T_analysys

path = 'Outputs'
filename_argument = click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='CSV_FORCE_FILE',
    )

dt_option = click.option(
    '--dt', 'dt',
    type=float,
    default=0.25,
    help='Timestep for the solute evolution given in fs and converted in a.u.',
    )

@maze.group()
def plot():
    pass

@plot.command()
@filename_argument
@dt_option
def force(filename, dt):
    plot_force(filename, dt)

@plot.command()
@filename_argument
@dt_option
def forcemod(filename, dt):
    plot_forcemod(filename, dt)
    
@plot.command()
@filename_argument
@dt_option
def temperature(filename, dt):
    PlotT(filename, dt)
