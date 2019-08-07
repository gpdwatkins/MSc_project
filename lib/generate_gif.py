import numpy as np
from copy import deepcopy
from os import listdir, unlink
from os.path import isfile, join
from lib.utils import *
from lib.generate_graphics import *
from datetime import datetime
from deepqlearning.dqn_agent import Agent
import torch
# from importlib import reload

# this file contains a function that generates a gif for a game of cat-and-mouse
# a policy can be provided (in the form of a .pth file containing weights and biases)
# if no policy is provided, the cat and mouse move randomly
# note that the policy refers only to the cat - the mouse continues to move randomly




# want to make this function general
# need to get it to accept a policy
# (in the form of a dict keyed on state_index or a .pth file of network weights)
# also want to it be able to play the game with different amounts of 'sight range'
# and also with perfect information, probability dist of mouse pos and no prob dist
# or should they be separate functions?

def play_cat_and_mouse(board_height, board_width, show_figs = True, policy = None, sight = float('inf'), use_belief_state = False, seed = None):
    # assumimg cat has sight 2 in each direction (i.e. can see a 5x5 grid around iteself)
    # cat and mouse move uniformly (can move any direction, or stay put, with prob 1/9)
    # cat policy doesn't update - stays uniform
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    if not seed is None:
        np.random.seed(seed)

    start_time = datetime.now().strftime('%Y%m%d_%H%M')
    gif_filename = 'graphics_gif/the_gif/output_' + start_time + '.gif'

    if policy == None:
        policy_type = 'random'
    elif type(policy) is str and policy[-4:] == '.txt':
        policy_type = 'state-action_dict'
        metadata = extract_training_metadata(policy)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        policy_dict = load_policy_from_file(policy)
    elif type(policy) is str and policy[-4:] == '.pth':
        policy_type = 'nn_weights'
        metadata = extract_training_metadata(policy)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        agent = Agent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
        agent.qnetwork_behaviour.load_state_dict(torch.load(policy))
    else:
        raise ValueError('Policy type not recognised. Should be None, dict or .pth filename')

    for file in listdir('graphics_gif'):
        file_with_path = join('graphics_gif', file)
        if isfile(file_with_path):
            unlink(file_with_path)

    filenames = []

    # Initialise stuff
    initial_board = np.array(initialise_board(board_height, board_width))
    cat_pos, mouse_pos = initialise_cat_mouse_positions(board_height, board_width)
    # use this line if I want to specify where the cat and mouse start
    # cat_pos, mouse_pos = (3,4), (3,5)
    # board = np.array(np.zeros([board_height, board_width]), dtype='O')
    # board[cat_pos] = 'C'
    # board[mouse_pos] = 'M'
    mouse_pos_prob_dist = initialise_mouse_prob_dist(board_height, board_width, cat_pos, mouse_pos, sight)

    print('Starting position')

    filename = 'gif_graphic_' + start_time + '_0' + '.png'
    filenames.append(filename)
    generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)

    iter = 1
    while cat_pos != mouse_pos:
        if policy_type == 'random':
            cat_vert_move = np.random.choice((-1,0,1))
            cat_horz_move = np.random.choice((-1,0,1))
        elif policy_type == 'state-action_dict':
            state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
            cat_action_index = policy_dict[state_index]
            cat_vert_move, cat_horz_move = action_index_to_moves(cat_action_index)
        elif policy_type == 'nn_weights':
            nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
            action_index = agent.act(nn_state)
            cat_vert_move, cat_horz_move = action_index_to_moves(action_index)
        else:
            raise ValueError('Never should have reached this point!')

        cat_move_stays_on_board = move_stays_on_board(cat_pos, cat_vert_move, cat_horz_move, board_height, board_width)
        cat_pos = (cat_pos[0] + cat_vert_move * cat_move_stays_on_board, cat_pos[1] + cat_horz_move * cat_move_stays_on_board)

        mouse_vert_move = np.random.choice((-1,0,1))
        mouse_horz_move = np.random.choice((-1,0,1))
        mouse_move_stays_on_board = move_stays_on_board(mouse_pos, mouse_vert_move, mouse_horz_move, board_height, board_width)
        mouse_pos = (mouse_pos[0] + mouse_vert_move * mouse_move_stays_on_board, mouse_pos[1] + mouse_horz_move * mouse_move_stays_on_board)

        # new_cat_pos = (cat_pos[0] + cat_vert_move, cat_pos[1] + cat_horz_move)
        # if (new_cat_pos[0] in range(board_height) and new_cat_pos[1] in range(board_width)):
        #     cat_pos = new_cat_pos
        # new_mouse_pos = (mouse_pos[0] + mouse_vert_move, mouse_pos[1] + mouse_horz_move)
        # if (new_mouse_pos[0] in range(board_height) and new_mouse_pos[1] in range(board_width)):
        #     mouse_pos = new_mouse_pos
        # board = np.array(np.zeros([board_height, board_width]), dtype='O')
        # board[cat_pos] = 'C'
        # board[mouse_pos] = 'M'

        if sight == float('inf'):
            # cat has perfect information of mouse position
            mouse_pos_prob_dist = np.zeros((board_height, board_width))
            mouse_pos_prob_dist[mouse_pos] = 1
        elif cat_can_see_mouse(cat_pos, mouse_pos, sight):
            mouse_pos_prob_dist = np.zeros((board_height, board_width))
            mouse_pos_prob_dist[mouse_pos] = 1
        elif not use_belief_state:
            new_mouse_pos_prob_dist = np.zeros((board_height, board_width))
            for row in range(board_height):
                for col in range(board_width):
                    if not ((abs(row - cat_pos[0]) <= sight) and (abs(col - cat_pos[1]) <= sight)):
                        new_mouse_pos_prob_dist[row, col] = 1
            sum_mouse_pos_prob_dist = np.sum(new_mouse_pos_prob_dist)
            new_mouse_pos_prob_dist /= sum_mouse_pos_prob_dist
            mouse_pos_prob_dist = deepcopy(new_mouse_pos_prob_dist)
        elif use_belief_state:
            new_mouse_pos_prob_dist = np.zeros((board_height, board_width))
            for row in range(board_height):
                for col in range(board_width):
                    for row_offset in [-1, 0, 1]:
                        for col_offset in [-1, 0, 1]:
                            if (((row + row_offset) in range(board_height)) and ((col + col_offset) in range(board_width))):
                                if not ((abs(row + row_offset - cat_pos[0]) <= sight) and (abs(col + col_offset - cat_pos[1]) <= sight)):
                                    new_mouse_pos_prob_dist[row + row_offset, col + col_offset] += (1/9) * mouse_pos_prob_dist[row,col]
                            else:
                                if not ((abs(row - cat_pos[0]) <= sight) and (abs(col - cat_pos[1]) <= sight)):
                                    new_mouse_pos_prob_dist[row, col] += (1/9) * mouse_pos_prob_dist[row,col]
            sum_mouse_pos_prob_dist = np.sum(new_mouse_pos_prob_dist)
            new_mouse_pos_prob_dist /= sum_mouse_pos_prob_dist
            mouse_pos_prob_dist = deepcopy(new_mouse_pos_prob_dist)

        print('\n Iteration %d' % iter)

        filename = 'gif_graphic_' + start_time + '_' + str(iter) + '.png'
        filenames.append(filename)

        generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)

        iter += 1

    generate_gif(filenames, gif_filename)
