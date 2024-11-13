import click

from ..input import initialize_from_yaml
from .maze import maze

# from .runners.main_Maze import main as main_Maze

# print('--------------------------------------------------------------------------------------')

# @maze.command()
# def thisisatest123():
#     print('This is a test123')

# print(maze)

@maze.group()
def run():
    pass

# @run.command()
# def thisisanothertest():
#     print('This is another test')

@run.command()
# @click.argument('N_p', type=int)
# @click.argument('N', type=int)
# @click.argument('N_steps', type=int)
# @click.argument('L', type=float)
def main_maze(N_p, N, N_steps, L):
    main_Maze(N_p, N, N_steps, L)

@run.command()
# @click.argument('N_p', type=int)
# @click.argument('N', type=int)
# @click.argument('N_steps', type=int)
# @click.argument('L', type=float)
@click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='YAML_INPUT_FILE',
    )
def main_maze_md(filename):
    gs, os, ms = initialize_from_yaml(filename)
    _main_maze_md(gs, os, ms)

def _main_maze_md(*args):
    from .runners.main_Maze_md import main as main_Maze_md
    main_Maze_md(*args)
