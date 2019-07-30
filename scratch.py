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

from lib.utils import *








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
# env_cliff = CliffWalkingEnv()
env = CatMouseEnv()

env.observation_space.n

print(env.reset())
env.render()

for action in range(9):
    print(env.step(action))
    env.render()


# board_height = 4
# board_width = 6
# actions = ['NW','N','NE','W','X','E','SW','S','SE']
# nS = board_height * board_height * board_width * board_width
# P = {}
# for s in range(nS):
#     (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(s, board_height, board_width)
# # for cat_vert_pos in range(board_height):
# #     for cat_horz_pos in range(board_width):
# #         for mouse_vert_pos in range(board_height):
# #             for mouse_horz_pos in range(board_width):
#     state_index = positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos), board_height, board_width)
#     P[state_index] = {}
#     for action in actions:
#         cat_vert_move, cat_horz_move = action_to_moves(action)
#         cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(board_height)) and ((cat_horz_pos + cat_horz_move) in range(board_width))
#         new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
#         new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
#         new_state_instances = {}
#         mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
#         for mouse_vert_move in mouse_vert_moves:
#             for mouse_horz_move in mouse_horz_moves:
#                 mouse_action = moves_to_action(mouse_vert_move, mouse_horz_move)
#                 mouse_move_stays_on_board = ((mouse_vert_pos + mouse_vert_move) in range(board_height)) and ((mouse_horz_pos + mouse_horz_move) in range(board_width))
#                 new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
#                 new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
#                 new_state = positions_to_state_index((new_cat_vert_pos, new_cat_horz_pos), (new_mouse_vert_pos, new_mouse_horz_pos), board_height, board_width)
#                 if not new_state in new_state_instances.keys():
#                     game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
#                     new_state_instances[new_state] = [1, game_over]
#                 else:
#                     new_state_instances[new_state][0] += 1
#                 # if (action == 'NW') and (cat_vert_pos == 5) and (cat_horz_pos == 0) and (mouse_vert_pos == 0) and (mouse_horz_pos == 3):
#                 #     print('mouse_action: (%d, %d)' % (mouse_vert_move, mouse_horz_move))
#                 #     print('new_mouse_location: (%d, %d)' % (new_mouse_vert_pos, new_mouse_horz_pos))
#         possible_next_states_list = []
#         for new_state, (no_instances, game_over) in new_state_instances.items():
#             possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, 1 * game_over, game_over)
#             possible_next_states_list.append(possible_next_state_tuple)
#         P[state_index][action_to_action_index(action)] = possible_next_states_list
#
# P[100]


# scratch_dict = {'one':1,'two':2,'three':3}
#
# scratch_dict['five'] = 5
# scratch_dict['six'] = 6
# scratch_dict['four'] = 4
# scratch_dict['other_four'] = 4
# scratch_dict['other_six'] = 6
#
# max_value = max(scratch_dict.values())
# best_actions = [action for action in scratch_dict.keys() if scratch_dict[action] == max_value]
#
# best_actions
#
# import random as rand
# rand.choice(best_actions)

stats_binary_reward
