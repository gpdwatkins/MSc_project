import numpy as np
from copy import deepcopy
from os import listdir, unlink
from os.path import isfile, join
from lib.utils import *
from lib.generate_graphics import *
from datetime import datetime
from deepqlearning.dqn_agent import DQNAgent
import torch
# from importlib import reload

# This file contains a function that generates a gif for a game of cat-and-mouse
# A policy can be provided as a string representing a filename -
# either a .txt file (which means it was trained using q learning)
# or a .pth file (which means it was trained using a deep neural network)
# If no policy is provided, the cat moves randomly
# Note that the policy refers only to the cat - the mouse always moves randomly


def play_cat_and_mouse(board_height, board_width, parameter_filename = None, sight = float('inf'), use_belief_state = False, seed = None, show_figs = True, walls = None):
    # mouse moves uniformly (can move any direction, or stay put, with prob 1/9)
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    if not seed is None:
        np.random.seed(seed)

    start_time = datetime.now().strftime('%Y%m%d_%H%M')
    gif_filename = 'graphics_gif/the_gif/output_' + start_time + '.gif'


    if parameter_filename == None:
        print("No parameter_filename provided; using random policy")
        policy_type = 'random'
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.txt':
        policy_type = 'state-action_dict'
        metadata = extract_training_metadata(parameter_filename)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        if (metadata['sight'] == 'None'):
            if not (sight is None):
                raise Exception('Policy was generated using different sight range')
        elif not float(metadata['sight']) == sight:
            raise Exception('Policy was generated using different sight range')
        if (metadata['walls'] == 'None'):
            if not walls == None:
                raise Exception('Inconsistent value for walls')
        # if not use_belief_state and not float(sight) == float('inf'):
            # raise Exception('Policy was generated using belief state')
        if parameter_filename.find('qlearning_qvalues/') == -1:
            parameter_filename = 'qlearning_qvalues/' + parameter_filename
            if parameter_filename.find('trained_parameters/') == -1:
                parameter_filename = 'trained_parameters/' + parameter_filename
        qvalues_dict = load_qvalues_from_file(parameter_filename)
        policy_dict = get_greedy_policy(qvalues_dict)
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.pth':
        metadata = extract_training_metadata(parameter_filename)
        if metadata['algorithm'] == 'dqn':
            policy_type = 'dqn_weights'
        elif metadata['algorithm'] == 'drqn':
            policy_type = 'drqn_weights'
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        if (metadata['sight'] == 'None'):
            if not (sight is None):
                raise Exception('Policy was generated using different sight range')
        elif not float(metadata['sight']) == sight:
            raise Exception('Policy was generated using different sight range')
        if not (metadata['use_belief_state'] == 'True') == use_belief_state:
            raise Exception('Inconsistent value for belief state')
        if (metadata['walls'] == 'None'):
            if not walls == None:
                raise Exception('Inconsistent value for walls')
        if metadata['algorithm'] == 'dqn' and parameter_filename.find('dqn_weights/') == -1:
            parameter_filename = 'dqn_weights/' + parameter_filename
        if metadata['algorithm'] == 'drqn' and parameter_filename.find('drqn_weights/') == -1:
            parameter_filename = 'drqn_weights/' + parameter_filename
        if parameter_filename.find('trained_parameters/') == -1:
            parameter_filename = 'trained_parameters/' + parameter_filename
        if policy_type == 'dqn_weights':
            agent = DQNAgent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
            agent.qnetwork_behaviour.load_state_dict(torch.load(parameter_filename))
        elif policy_type == 'drqn_weights':
            agent = DRQNAgent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
            agent.drqn_behaviour.load_state_dict(torch.load(parameter_filename))
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
    if policy_type == 'drqn_weights':
        hidden = agent.init_hidden()
    # use this line if I want to specify where the cat and mouse start
    # cat_pos, mouse_pos = (3,4), (3,5)
    mouse_pos_prob_dist = initialise_mouse_prob_dist(board_height, board_width, cat_pos, mouse_pos, sight, walls = walls)

    print('Starting position')

    filename = 'gif_graphic_' + start_time + '_0' + '.png'
    filenames.append(filename)
    generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs, walls)

    iter = 1
    while cat_pos != mouse_pos:
        if policy_type == 'random':
            cat_vert_move = np.random.choice((-1,0,1))
            cat_horz_move = np.random.choice((-1,0,1))
        elif policy_type == 'state-action_dict':
            state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
            cat_action_index = policy_dict[state_index]
            cat_vert_move, cat_horz_move = action_index_to_moves(cat_action_index)
        elif policy_type == 'dqn_weights':
            nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
            action_index = agent.act(nn_state)
            cat_vert_move, cat_horz_move = action_index_to_moves(action_index)
        elif policy_type == 'drqn_weights':
            nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
            drqn_state = nn_input_as_drqn_input(board_height, board_width, nn_state)
            action_index, hidden = agent.act(drqn_state, hidden)
            cat_vert_move, cat_horz_move = action_index_to_moves(action_index)
        else:
            raise ValueError('Never should have reached this point!')

        cat_move_stays_on_board = move_is_legal(cat_pos, cat_vert_move, cat_horz_move, board_height, board_width, walls = walls)
        cat_pos = (cat_pos[0] + cat_vert_move * cat_move_stays_on_board, cat_pos[1] + cat_horz_move * cat_move_stays_on_board)

        mouse_vert_move = np.random.choice((-1,0,1))
        mouse_horz_move = np.random.choice((-1,0,1))
        mouse_move_stays_on_board = move_is_legal(mouse_pos, mouse_vert_move, mouse_horz_move, board_height, board_width, walls = walls)
        mouse_pos = (mouse_pos[0] + mouse_vert_move * mouse_move_stays_on_board, mouse_pos[1] + mouse_horz_move * mouse_move_stays_on_board)

        if sight == None or cat_can_see_mouse(cat_pos, mouse_pos, sight, walls = walls):
            mouse_pos_prob_dist = np.zeros((board_height, board_width))
            mouse_pos_prob_dist[mouse_pos] = 1
        elif not use_belief_state:
            new_mouse_pos_prob_dist = np.zeros((board_height, board_width))
            for row in range(board_height):
                for col in range(board_width):
                    if not cat_can_see_mouse(cat_pos, (row, col), sight = sight, walls = walls):
                    # ((abs(row - cat_pos[0]) <= sight) and (abs(col - cat_pos[1]) <= sight)):
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
                            if move_is_legal((row, col), row_offset, col_offset, board_height, board_width, walls = walls):
                            # (((row + row_offset) in range(board_height)) and ((col + col_offset) in range(board_width))):
                                if not cat_can_see_mouse(cat_pos, (row + row_offset, col + col_offset), sight = sight, walls = walls):
                                # ((abs(row + row_offset - cat_pos[0]) <= sight) and (abs(col + col_offset - cat_pos[1]) <= sight)):
                                    new_mouse_pos_prob_dist[row + row_offset, col + col_offset] += (1/9) * mouse_pos_prob_dist[row,col]
                            else:
                                if not cat_can_see_mouse(cat_pos, (row, col), sight = sight, walls = walls):
                                # ((abs(row - cat_pos[0]) <= sight) and (abs(col - cat_pos[1]) <= sight)):
                                    new_mouse_pos_prob_dist[row, col] += (1/9) * mouse_pos_prob_dist[row,col]
            sum_mouse_pos_prob_dist = np.sum(new_mouse_pos_prob_dist)
            new_mouse_pos_prob_dist /= sum_mouse_pos_prob_dist
            mouse_pos_prob_dist = deepcopy(new_mouse_pos_prob_dist)

        print('\n Iteration %d' % iter)

        filename = 'gif_graphic_' + start_time + '_' + str(iter) + '.png'
        filenames.append(filename)

        generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs, walls)

        iter += 1

    generate_gif(filenames, gif_filename)
