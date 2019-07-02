import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import copy
# import scipy as sp
# import matplotlib as mpl
# import matplotlib.image as image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import imageio
# from os import listdir, unlink
# from os.path import isfile, join
# import xarray as xr
# import gym










def run_q_learning(board_height, board_width):
    # initiate_board
    # environment completely observable
    # so state is (cat_position, mouse_position, map_configuration)
    # tensor is of form: cat_vert_pos x cat_horz_pos mouse_vert_pos mouse_horz_pos cat_action
    # figure out how to use tensorflow for this?
    # Next step is to replace the direct calculation and storing of q values with a NN
    # want a reset function and a step function
    # reset: input: none; output: state (the initial state which isn't necessarily the same every time)
    # step: input: action; output: state (the new state),

    # initialise map
    initial_board = np.array(initialise_board(board_height, board_width))

    # randomly generate cat, mouse starting positions
    cat_pos, mouse_pos = initialise_cat_mouse_positions(initial_board)

    # the states are the possible positions of the cat and mouse
    # in the form ((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos)
    # they are ordered and enumerated
    # i.e. there are board_height * board_width * board_height * board_width states
    state = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)

    # the actions are ['NW','N','NE','W','X','E','SW','S','SE']


    # initial_q_values = np.zeros([board_height, board_width, board_height, board_width, 9])
    # cat_rows = range(board_height)
    # cat_cols = range(board_width)
    # mouse_rows = range(board_height)
    # mouse_cols = range(board_width)
    # moves = ['NW','N','NE','W','X','E','SW','S','SE']

    # q_values = xr.DataArray(initial_q_values, coords=[cat_rows, cat_cols, mouse_rows, mouse_cols, moves], dims=['cat_row', 'cat_col', 'mouse_row', 'cat_row', 'move'])

from cat_mouse_env import CatMouseEnv
from cliffwalking import CliffWalkingEnv
env_cliff = CliffWalkingEnv()
env = CatMouseEnv()

print(env.reset())
env.render()

for action in range(9):
    print(env.step(action))
    env.render()
