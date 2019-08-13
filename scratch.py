import numpy as np
from collections import namedtuple

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


def generate_transition_probabilities(board_height, board_width, current_state, action_index, sight = 2, mouse_pos_dist = True):
    # current state is defined as:
    # spaces.Box(low=0, high=1.0, shape=(2 * board_height * board_width), dtype=np.float32)
    # the first (board_height * board_width) dimensions refer to the cat's position
    # they contain the cat position as a one-hot encoding
    # the last (board_height * board_width) dimensions refer to the mouse's position
    # they contain the  probabliity that the mouse is in each square

    cat_state = current_state[:board_height * board_width]
    mouse_state = current_state[board_height * board_width:]
    cat_pos_index = list(cat_state).index(1)
    cat_vert_pos, cat_horz_pos = board_pos_index_to_board_pos(cat_pos_index, board_height, board_width)
    cat_vert_moves, cat_horz_moves = action_index_to_moves(action_index)
    cat_move_stays_on_board = move_stays_on_board((cat_vert_pos, cat_horz_pos), cat_vert_move, cat_horz_move, board_height, board_width)
    new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
    new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board


    for mouse_pos_index in range(board_height * board_width):
        (mouse_vert_pos, mouse_horz_pos) = board_pos_index_to_board_pos(mouse_pos_index, board_height, board_width)
        state_index = self._positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos))
        P[state_index] = {}
        # for action in actions:
        #     cat_vert_move, cat_horz_move = self._action_to_moves(action)
        #     cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(self.board_height)) and ((cat_horz_pos + cat_horz_move) in range(self.board_width))
        #     new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
        #     new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
        #     new_state_instances = {}
        mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
        for mouse_vert_move in mouse_vert_moves:
            for mouse_horz_move in mouse_horz_moves:
                # mouse_action = self._moves_to_action(mouse_vert_move, mouse_horz_move)
                mouse_move_stays_on_board = move_stays_on_board((mouse_vert_pos, mouse_horz_pos), mouse_vert_move, mouse_horz_move, self.board_height, self.board_width)
                new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
                new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
                new_mouse_pos_index = board_pos_to_board_pos_index((mouse_vert_pos, mouse_horz_pos), board_height, board_width)
                if not new_state in new_state_instances.keys():
                    game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
                    new_state_instances[new_state] = [1, game_over]
                else:
                    new_state_instances[new_state][0] += 1
            possible_next_states_list = []
            for new_state, (no_instances, game_over) in new_state_instances.items():
                possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, 1 * game_over, game_over)
                possible_next_states_list.append(possible_next_state_tuple)
            P[state_index][self._action_to_action_index(action)] = possible_next_states_list











def observed_state_as_nn_input(board_height, board_width, true_state_index, previous_observed_state_nn_input, sight = 2, use_belief_state = True):
    # anything that has the form of a nn_input will be a list of length (2 * board_height * board_width)
    # it contains the cat position as a one-hot encoding (as it is known by the cat)
    # and a probability distribution for the mouse position

    # cat_state_one_hot = previous_observed_state_nn_input[:board_height * board_width]

    # cat_pos_index = list(cat_state_one_hot).index(1)
    # cat_vert_pos, cat_horz_pos = board_pos_index_to_board_pos(cat_pos_index, board_height, board_width)
    true_cat_pos, true_mouse_pos = state_index_to_positions(true_state_index, board_height, board_width)
    true_cat_pos_index = board_pos_to_board_pos_index(true_cat_pos, board_height, board_width)
    true_mouse_pos_index = board_pos_to_board_pos_index(true_mouse_pos, board_height, board_width)

    cat_pos_one_hot = np.zeros(board_height * board_width)
    mouse_pos_dist = np.zeros(board_height * board_width)
    cat_pos_one_hot[true_cat_pos_index] = 1

    if cat_can_see_mouse(true_cat_pos, true_mouse_pos, sight):
        mouse_pos_dist[true_mouse_pos_index] = 1
    elif use_belief_state == False:
        total_unseen_positions = 0
        for row in range(board_height):
            for col in range(board_width):
                mouse_board_pos = (row, col)
                mouse_board_pos_index = board_pos_to_board_pos_index(mouse_board_pos, board_height, board_width)
                # board_pos_index = board_pos_to_board_pos_index(board_pos, board_height, board_width)
                if not cat_can_see_mouse(true_cat_pos, mouse_board_pos, sight):
                    mouse_pos_dist[mouse_board_pos_index] = 1
                    total_unseen_positions += 1
        mouse_pos_dist = mouse_pos_dist/total_unseen_positions
    else:
        previous_mouse_pos_dist = previous_observed_state_nn_input[board_height * board_width:]
        mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
        for previous_mouse_row in range(board_height):
            for previous_mouse_col in range(board_width):
                previous_mouse_pos = (previous_mouse_row, previous_mouse_col)
                previous_mouse_pos_index = board_pos_to_board_pos_index(previous_mouse_pos, board_height, board_width)
                for mouse_vert_move in mouse_vert_moves:
                    for mouse_horz_move in mouse_horz_moves:
                        mouse_move_stays_on_board = move_stays_on_board(previous_mouse_pos, mouse_vert_move, mouse_horz_move, board_height, board_width)
                        new_mouse_vert_pos = previous_mouse_row + mouse_vert_move * mouse_move_stays_on_board
                        new_mouse_horz_pos = previous_mouse_col + mouse_horz_move * mouse_move_stays_on_board
                        new_mouse_pos = (new_mouse_vert_pos, new_mouse_horz_pos)
                        new_mouse_pos_index = board_pos_to_board_pos_index(new_mouse_pos, board_height, board_width)
                        if not cat_can_see_mouse(true_cat_pos, new_mouse_pos, sight):
                            mouse_pos_dist[new_mouse_pos_index] += (1/9) * previous_mouse_pos_dist[previous_mouse_pos_index]
        mouse_pos_dist = mouse_pos_dist/sum(mouse_pos_dist)

    cat_output = np.array([list(np.zeros(board_width)) for row in range(board_height)])
    for index, prob in enumerate(cat_pos_one_hot):
        row, col = board_pos_index_to_board_pos(index, board_height, board_width)
        cat_output[row][col] = prob
    print(cat_output)

    mouse_output = np.array([list(np.zeros(board_width)) for row in range(board_height)])
    for index, prob in enumerate(mouse_pos_dist):
        row, col = board_pos_index_to_board_pos(index, board_height, board_width)
        mouse_output[row][col] = prob
    print(mouse_output)

    return np.concatenate((cat_pos_one_hot, mouse_pos_dist))




# Need to test this now ^^^^^^^^^^^^^^^^^
# will need to construct some inputs
# for this, maybe need to change the functions in utils.py that convert from/to nn_input
# as it currently encodes the cat_pos using 2 datapoints (row and height) rather than one-hot

board_height = 4
board_width = 6
cat_start_pos = (0,2)
mouse_start_pos = (3,1)
sight = 1
use_belief_state = True
true_state_index = positions_to_state_index(cat_start_pos, mouse_start_pos, board_height, board_width)
# initial_cat_distribution = np.zeros(board_height * board_width)
# initial_mouse_distribution = np.ones(board_height * board_width)
# * (1/(board_height * board_width))
initial_state_distribution = np.ones(2 * board_height * board_width)
# np.concatenate((initial_cat_distribution, initial_mouse_distribution))
nn_input = observed_state_as_nn_input(board_height, board_width, true_state_index, initial_state_distribution, sight = sight, use_belief_state = use_belief_state)

new_cat_pos = (1,2)
new_mouse_pos = (3,1)
true_state_index = positions_to_state_index(new_cat_pos, new_mouse_pos, board_height, board_width)
nn_input = observed_state_as_nn_input(board_height, board_width, true_state_index, nn_input, sight = sight, use_belief_state = use_belief_state)

new_cat_pos = (0,2)
new_mouse_pos = (3,1)
true_state_index = positions_to_state_index(new_cat_pos, new_mouse_pos, board_height, board_width)
nn_input = observed_state_as_nn_input(board_height, board_width, true_state_index, nn_input, sight = sight, use_belief_state = use_belief_state)

new_cat_pos = (1,3)
new_mouse_pos = (2,0)
true_state_index = positions_to_state_index(new_cat_pos, new_mouse_pos, board_height, board_width)
nn_input = observed_state_as_nn_input(board_height, board_width, true_state_index, nn_input, sight = sight, use_belief_state = use_belief_state)

new_cat_pos = (2,2)
new_mouse_pos = (1,1)
true_state_index = positions_to_state_index(new_cat_pos, new_mouse_pos, board_height, board_width)
nn_input = observed_state_as_nn_input(board_height, board_width, true_state_index, nn_input, sight = sight, use_belief_state = use_belief_state)



scratch = np.arange(6)
scratch
np.reshape(scratch,(2,3))



import json

dict = {'Python' : '.py', 'C++' : '.cpp', 'Java' : '.java'}

json = json.dumps(dict)
f = open("dict.json","w")
f.write(json)
f.close()


myfile = 'dict.txt'
with open(myfile, 'w') as f:
    for key, value in dict.items():
        f.write('%s:%s\n' % (key, value))

data = {}
with open(myfile) as raw_data:
    for item in raw_data:
        if ':' in item:
            key,value = item.split(':', 1)
            key = key.rstrip()
            value = value.rstrip()
            # print(value[-2:])
            # if key[-2:] == '\n':
            #     print('here1')
            #     key = key[:-2]
            # if value[-2:] == '\n':
            #     value = value[:-2]
            #     print('here2')
            data[key]=value
        else:
            pass # deal with bad lines of text here

data




def test_policy(env, board_height, board_width, no_episodes, policy = None, seed = None):
    # assumimg cat has sight 2 in each direction (i.e. can see a 5x5 grid around iteself)
    # cat and mouse move uniformly (can move any direction, or stay put, with prob 1/9)
    # cat policy doesn't update - stays uniform
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    if not seed is None:
        np.random.seed(seed)

    # start_time = datetime.now().strftime('%Y%m%d_%H%M')
    # gif_filename = 'graphics_gif/the_gif/output_' + start_time + '.gif'

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

    # for file in listdir('graphics_gif'):
    #     file_with_path = join('graphics_gif', file)
    #     if isfile(file_with_path):
    #         unlink(file_with_path)
    #
    # filenames = []

    stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
        episode_lengths=np.zeros(no_episodes),
        episode_rewards=np.zeros(no_episodes))

    for episode in range(1, no_episodes+1):

        # Initialise stuff
        initial_board = np.array(initialise_board(board_height, board_width))
        cat_pos, mouse_pos = initialise_cat_mouse_positions(board_height, board_width)
        # use this line if I want to specify where the cat and mouse start
        # cat_pos, mouse_pos = (3,4), (3,5)
        # board = np.array(np.zeros([board_height, board_width]), dtype='O')
        # board[cat_pos] = 'C'
        # board[mouse_pos] = 'M'
        # mouse_pos_prob_dist = initialise_mouse_prob_dist(board_height, board_width, cat_pos, mouse_pos, sight)

        # print('Starting position')

        # filename = 'gif_graphic_' + start_time + '_0' + '.png'
        # filenames.append(filename)
        # generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)


        for timestep in itertools.count():
            if policy_type == 'random':
                action_index = np.random.randint(9)
                # cat_vert_move = np.random.choice((-1,0,1))
                # cat_horz_move = np.random.choice((-1,0,1))
            elif policy_type == 'state-action_dict':
                state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
                cat_action_index = policy_dict[state_index]
                # cat_vert_move, cat_horz_move = action_index_to_moves(cat_action_index)
            elif policy_type == 'nn_weights':
                nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
                action_index = agent.act(nn_state)
                # cat_vert_move, cat_horz_move = action_index_to_moves(action_index)
            else:
                raise ValueError('Never should have reached this point!')



            # Take a step
            # action_probs = behaviour_policy(current_state)
            # action = np.random.choice(range(len(action_probs)), p=list(action_probs.values()))
            next_state, reward, done, _ = env.step(action_index)

            # Update statistics
            stats.episode_rewards[episode-1] += reward

            # TD Update
            # max_q_value = max(Q[next_state].values())
            # best_next_actions = [action for action in Q[next_state].keys() if Q[next_state][action] == max_q_value]
            # best_next_action = rand.choice(best_next_actions)
            # td_target = reward + discount_factor * Q[next_state][best_next_action]
            # td_delta = td_target - Q[current_state][action]
            # Q[current_state][action] += alpha * td_delta

            if done:
                stats.episode_lengths[episode-1] = timestep
                break



        # iter = 1
        # while cat_pos != mouse_pos:
        #     if policy_type == 'random':
        #         cat_vert_move = np.random.choice((-1,0,1))
        #         cat_horz_move = np.random.choice((-1,0,1))
        #     elif policy_type == 'state-action_dict':
        #         state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
        #         cat_action_index = policy_dict[state_index]
        #         cat_vert_move, cat_horz_move = action_index_to_moves(cat_action_index)
        #     elif policy_type == 'nn_weights':
        #         nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
        #         action_index = agent.act(nn_state)
        #         cat_vert_move, cat_horz_move = action_index_to_moves(action_index)
        #     else:
        #         raise ValueError('Never should have reached this point!')
        #
        #     cat_move_stays_on_board = move_stays_on_board(cat_pos, cat_vert_move, cat_horz_move, board_height, board_width)
        #     cat_pos = (cat_pos[0] + cat_vert_move * cat_move_stays_on_board, cat_pos[1] + cat_horz_move * cat_move_stays_on_board)
        #
        #     mouse_vert_move = np.random.choice((-1,0,1))
        #     mouse_horz_move = np.random.choice((-1,0,1))
        #     mouse_move_stays_on_board = move_stays_on_board(mouse_pos, mouse_vert_move, mouse_horz_move, board_height, board_width)
        #     mouse_pos = (mouse_pos[0] + mouse_vert_move * mouse_move_stays_on_board, mouse_pos[1] + mouse_horz_move * mouse_move_stays_on_board)



            # mouse_vert_move = np.random.choice((-1,0,1))
            # mouse_horz_move = np.random.choice((-1,0,1))
            # new_cat_pos = (cat_pos[0] + cat_vert_move, cat_pos[1] + cat_horz_move)
            # if (new_cat_pos[0] in range(board_height) and new_cat_pos[1] in range(board_width)):
            #     cat_pos = new_cat_pos
            # new_mouse_pos = (mouse_pos[0] + mouse_vert_move, mouse_pos[1] + mouse_horz_move)
            # if (new_mouse_pos[0] in range(board_height) and new_mouse_pos[1] in range(board_width)):
            #     mouse_pos = new_mouse_pos
            # board = np.array(np.zeros([board_height, board_width]), dtype='O')
            # board[cat_pos] = 'C'
            # board[mouse_pos] = 'M'

            # if sight == float('inf'):
            #     # cat has perfect information of mouse position
            #     mouse_pos_prob_dist = np.zeros((board_height, board_width))
            #     mouse_pos_prob_dist[mouse_pos] = 1
            # elif cat_can_see_mouse(cat_pos, mouse_pos, sight):
            #     mouse_pos_prob_dist = np.zeros((board_height, board_width))
            #     mouse_pos_prob_dist[mouse_pos] = 1
            # elif not use_belief_state:
            #     new_mouse_pos_prob_dist = np.zeros((board_height, board_width))
            #     for row in range(board_height):
            #         for col in range(board_width):
            #             if not ((abs(row - cat_pos[0]) <= sight) and (abs(col - cat_pos[1]) <= sight)):
            #                 new_mouse_pos_prob_dist[row, col] = 1
            #     sum_mouse_pos_prob_dist = np.sum(new_mouse_pos_prob_dist)
            #     new_mouse_pos_prob_dist /= sum_mouse_pos_prob_dist
            #     mouse_pos_prob_dist = deepcopy(new_mouse_pos_prob_dist)
            # elif use_belief_state:
            #     new_mouse_pos_prob_dist = np.zeros((board_height, board_width))
            #     for row in range(board_height):
            #         for col in range(board_width):
            #             for row_offset in [-1, 0, 1]:
            #                 for col_offset in [-1, 0, 1]:
            #                     if (((row + row_offset) in range(board_height)) and ((col + col_offset) in range(board_width))):
            #                         if not ((abs(row + row_offset - cat_pos[0]) <= sight) and (abs(col + col_offset - cat_pos[1]) <= sight)):
            #                             new_mouse_pos_prob_dist[row + row_offset, col + col_offset] += (1/9) * mouse_pos_prob_dist[row,col]
            #                     else:
            #                         if not ((abs(row - cat_pos[0]) <= sight) and (abs(col - cat_pos[1]) <= sight)):
            #                             new_mouse_pos_prob_dist[row, col] += (1/9) * mouse_pos_prob_dist[row,col]
            #     sum_mouse_pos_prob_dist = np.sum(new_mouse_pos_prob_dist)
            #     new_mouse_pos_prob_dist /= sum_mouse_pos_prob_dist
            #     mouse_pos_prob_dist = deepcopy(new_mouse_pos_prob_dist)
            #
            # print('\n Iteration %d' % iter)
            #
            # filename = 'gif_graphic_' + start_time + '_' + str(iter) + '.png'
            # filenames.append(filename)
            #
            # generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)

            # iter += 1

    # generate_gif(filenames, gif_filename)


    return np.average(stats.episode_lengths), np.average(stats.episode_rewards)
