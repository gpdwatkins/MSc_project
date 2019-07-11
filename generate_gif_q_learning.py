import numpy as np
from copy import deepcopy
from os import listdir, unlink
from os.path import isfile, join
from lib.utils import *
from lib.generate_graphics import *
# from lib import plotting
from datetime import datetime
from q_learning import *
from cat_mouse_env import CatMouseEnv_binary_reward, CatMouseEnv_proximity_reward
from importlib import reload

# import lib.generate_graphics
# reload(lib.generate_graphics)

def play_cat_and_mouse_q_learning(cat_policy, board_height, board_width, show_figs = True):
    # assumimg cat has sight 2 in each direction (i.e. can see a 5x5 grid around iteself)
    # cat and mouse move uniformly (can move any direction, or stay put, with prob 1/9)
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    if not len(cat_policy) == board_height * board_width * board_height * board_width:
        raise Exception('Policy incompatible with given board size')

    start_time = datetime.now().strftime('%Y%m%d_%H%M')

    for file in listdir('graphics_gif'):
        file_with_path = join('graphics_gif', file)
        if isfile(file_with_path):
            unlink(file_with_path)

    filenames = []

    # Initialise stuff
    initial_board = np.array(initialise_board(board_height, board_width))

    cat_pos, mouse_pos = initialise_cat_mouse_positions(board_height, board_width)


    board = np.array(np.zeros([board_height, board_width]), dtype='O')
    board[cat_pos] = 'C'
    board[mouse_pos] = 'M'

    # mouse_pos_prob_dist = initialise_mouse_prob_dist(board_height, board_width, cat_pos, mouse_pos)
    mouse_pos_prob_dist = np.zeros((board_height, board_width))
    mouse_pos_prob_dist[mouse_pos] = 1

    print('Starting position')

    filename = 'gif_graphic_' + start_time + '_0' + '.png'
    filenames.append(filename)
    generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)

    iter = 1
    while cat_pos != mouse_pos:
        state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
        cat_action_index = cat_policy[state_index]
        cat_vert_move, cat_horz_move = action_index_to_moves(cat_action_index)
        mouse_vert_move = np.random.choice((-1,0,1))
        mouse_horz_move = np.random.choice((-1,0,1))
        new_cat_pos = (cat_pos[0] + cat_vert_move, cat_pos[1] + cat_horz_move)
        if (new_cat_pos[0] in range(board_height) and new_cat_pos[1] in range(board_width)):
            cat_pos = new_cat_pos
        new_mouse_pos = (mouse_pos[0] + mouse_vert_move, mouse_pos[1] + mouse_horz_move)
        if (new_mouse_pos[0] in range(board_height) and new_mouse_pos[1] in range(board_width)):
            mouse_pos = new_mouse_pos
        board = np.array(np.zeros([board_height, board_width]), dtype='O')
        board[cat_pos] = 'C'
        board[mouse_pos] = 'M'

        # fully observable so cat can always see mouse
        mouse_pos_prob_dist = np.zeros((board_height, board_width))
        mouse_pos_prob_dist[mouse_pos] = 1

        print('\n Iteration %d' % iter)

        filename = 'gif_graphic_' + start_time + '_' + str(iter) + '.png'
        filenames.append(filename)

        gif_filename = 'graphics_gif/the_gif/output_' + start_time + '.gif'

        generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)
        iter += 1

    generate_gif(filenames, gif_filename)

board_height = 4
board_width = 6

Q_binary_reward, stats_binary_reward = q_learning(CatMouseEnv_binary_reward(board_height, board_width), 100000)
plot_episode_stats(stats_binary_reward, 1000)


cat_policy_binary_reward = get_greedy_policy(Q_binary_reward)

play_cat_and_mouse_q_learning(cat_policy_binary_reward, board_height, board_width)

show_policy(cat_policy_binary_reward, (1,3))


Q_proximity_reward, stats_proximity_reward = q_learning(CatMouseEnv_proximity_reward(board_height, board_width), 100000)
plot_episode_stats(stats_proximity_reward, 1000)


cat_policy_proximity_reward = get_greedy_policy(Q_proximity_reward)

play_cat_and_mouse_q_learning(cat_policy_proximity_reward, board_height, board_width)

show_policy(cat_policy_proximity_reward, (0,5))
