import numpy as np
from collections import namedtuple
import torch
from deepqlearning.dqn_agent import DQNAgent

def initialise_board(height=19, width=19, walls=False):
    if not walls:
        return np.zeros((height,width))
    else:
        return np.round(np.random.rand(height,width))


def initialise_cat_mouse_positions(board_height, board_width):
    # board_height, board_width = np.shape(board)
    cat_pos = (np.random.randint(0,board_height), np.random.randint(0,board_width))
    mouse_pos = (np.random.randint(0,board_height), np.random.randint(0,board_width))
    while mouse_pos == cat_pos:
        mouse_pos = (np.random.randint(0,board_height), np.random.randint(0,board_width))
    return cat_pos, mouse_pos


def cat_can_see_mouse(cat_pos, mouse_pos, sight, walls):
    # assumimg cat has can see sight squares in each direction (i.e. can see a (1+2xsight)x(1+2xsight) grid around itself)
    if sight == None:
    # sight = None means perfect information
        return True

    if not ((abs(cat_pos[0]-mouse_pos[0]) <= sight) and (abs(cat_pos[1]-mouse_pos[1]) <= sight)):
        return False

    if walls == None:
        return True

    if cat_pos[1] - mouse_pos[1] == 0:
        def line_of_sight_get_col(cat_pos, row):
            return cat_pos[1]
    # elif cat_pos[0] - mouse_pos[0] == 0:
    #     def line_of_sight_get_row(cat_pos, col):
    #         return cat_pos[0]
    else:
        line_of_sight_grad = (cat_pos[0] - mouse_pos[0]) / (cat_pos[1] - mouse_pos[1])
        def line_of_sight_get_row(cat_pos, col):
            return line_of_sight_grad * (col - cat_pos[1]) + cat_pos[0]
        def line_of_sight_get_col(cat_pos, row):
            return (1/line_of_sight_grad) * (row - cat_pos[0]) + cat_pos[1]

    for row_index in range(min(cat_pos[0], mouse_pos[0]), max(cat_pos[0], mouse_pos[0])):
        col_index = line_of_sight_get_col(cat_pos, row_index + 0.5)
        # print(row_index + 0.5, col_index)
        if (col_index + 0.5) % 1 == 0:
            if (row_index + 0.5, col_index - 0.5) in walls or\
            (row_index + 0.5, col_index + 0.5) in walls:
                return False
        else:
            if (row_index + 0.5, round(col_index)) in walls:
                return False

    for col_index in range(min(cat_pos[1], mouse_pos[1]), max(cat_pos[1], mouse_pos[1])):
        row_index = line_of_sight_get_row(cat_pos, col_index + 0.5)
        # print(row_index, col_index + 0.5)
        if (row_index - 0.5) % 1 == 0:
            if (row_index - 0.5, col_index + 0.5) in walls or\
            (row_index + 0.5, col_index + 0.5) in walls:
                return False
        else:
            if (round(row_index), col_index + 0.5) in walls:
                return False
    return True


def initialise_mouse_prob_dist(board_height, board_width, cat_pos, mouse_pos, sight, walls = None):
    if cat_can_see_mouse(cat_pos, mouse_pos, sight, walls = walls):
        mouse_pos_prob_dist = np.zeros((board_height, board_width))
        mouse_pos_prob_dist[mouse_pos] = 1
    else:
        mouse_pos_prob_dist = np.ones((board_height, board_width))
        no_observable_cells = 0
        for row_offset in range(-int(min(sight,board_height)),int(min(sight,board_height))+1):
            for col_offset in range(-int(min(sight,board_width)),int(min(sight,board_width))+1):
                if ((cat_pos[0] + row_offset in range(board_height)) and (cat_pos[1] + col_offset in range(board_width))):
                    if cat_can_see_mouse(cat_pos, (cat_pos[0] + row_offset, cat_pos[1] + col_offset), sight = sight, walls = walls):
                        no_observable_cells += 1
                        mouse_pos_prob_dist[cat_pos[0] + row_offset, cat_pos[1] + col_offset] = 0
        mouse_pos_prob_dist *= 1/(board_height * board_width - no_observable_cells)
    return mouse_pos_prob_dist


def board_pos_index_to_board_pos(board_pos_index, board_height, board_width):
    if not board_pos_index in range(board_height * board_width):
        raise ValueError('board_pos_index must be between 0 and %d' % (board_height * board_width - 1))

    board_row = float(np.mod(board_pos_index,board_height))
    board_col = np.mod(board_pos_index - board_row, board_height * board_width) / board_height
    return (int(board_row), int(board_col))


def board_pos_to_board_pos_index(board_pos, board_height, board_width):
    if not np.prod([ \
    board_pos[0] in range(board_height), \
    board_pos[1] in range(board_width), \
    ]):
        raise ValueError('Board position (%d,%d) is outside the board' % (board_pos[0], board_pos[1]))

    board_pos_index = int( \
    board_pos[0] + \
    board_pos[1] * board_height \
    )
    return board_pos_index


def state_index_to_positions(state_index, board_height, board_width):
    if not state_index in range(board_height**2 * board_width**2):
        raise ValueError('state_index must be between 0 and %d' % (board_height**2 * board_width**2 - 1))

    cat_vert_pos = float(np.mod(state_index,board_height))
    cat_horz_pos = np.mod(state_index - cat_vert_pos, board_height * board_width) / board_height
    mouse_vert_pos = np.mod(state_index - cat_horz_pos * board_height - cat_vert_pos, board_height * board_width * board_height) / (board_height * board_width)
    mouse_horz_pos = (state_index - mouse_vert_pos * board_height * board_width - cat_horz_pos * board_height - cat_vert_pos) / (board_height * board_width * board_height)
    return (int(cat_vert_pos), int(cat_horz_pos)), (int(mouse_vert_pos), int(mouse_horz_pos))


def positions_to_state_index(cat_pos, mouse_pos, board_height, board_width):
    if not np.prod([ \
    cat_pos[0] in range(board_height), \
    cat_pos[1] in range(board_width), \
    ]):
        raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))

    if not np.prod([ \
    mouse_pos[0] in range(board_height), \
    mouse_pos[1] in range(board_width) \
    ]):
        raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))

    return int( \
    cat_pos[0] + \
    cat_pos[1] * board_height  + \
    mouse_pos[0] * board_height * board_width + \
    mouse_pos[1] * board_height * board_width * board_height \
    )


def positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width):
    # nn_input combines elements representing cat position and mouse position
    # the first two elements are the row and column of the cat position
    # the next (size board_height x board_width) elements represent the mouse position
    # each square on the board is represented by an element
    # if the cat has perfect information (i.e. knows where the mouse is), this is a one-hot encoding
    # if the cat knows a probability distribution, the vector contains this distribution

    # note that we only use this function if the cat and mouse positions are known
    # the mouse prob dist is therefore effectively a one-hot encoding
    if not np.prod([ \
    cat_pos[0] in range(board_height), \
    cat_pos[1] in range(board_width), \
    ]):
        raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))

    if not np.prod([ \
    mouse_pos[0] in range(board_height), \
    mouse_pos[1] in range(board_width) \
    ]):
        raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))

    cat_pos_one_hot = np.zeros(board_height * board_width)
    cat_pos_index = board_pos_to_board_pos_index(cat_pos, board_height, board_width)
    cat_pos_one_hot[cat_pos_index] = 1

    mouse_pos_dist = np.zeros(board_height * board_width)
    mouse_pos_index = board_pos_to_board_pos_index(mouse_pos, board_height, board_width)
    mouse_pos_dist[mouse_pos_index] = 1

    output_vector = np.concatenate((cat_pos_one_hot, mouse_pos_dist))
    return output_vector


def nn_input_to_positions(nn_input, board_height, board_width):
    # this function assumes perfect information
    # so the mouse distribution is effectively a one-hot encoding
    if not len(nn_input) == 2 * board_height * board_width:
        raise ValueError('nn_input vector has wrong dimensions')

    # cat_vert_pos = nn_input[0] * (board_height-1)
    # cat_horz_pos = nn_input[1] * (board_width-1)
    cat_pos_index = list(nn_input[:board_height * board_width]).index(1)
    cat_vert_pos, cat_horz_pos = board_pos_index_to_board_pos(cat_pos_index, board_height, board_width)

    mouse_pos_index = list(nn_input[board_height * board_width:]).index(1)
    mouse_vert_pos, mouse_horz_pos = board_pos_index_to_board_pos(mouse_pos_index, board_height, board_width)
    return (int(cat_vert_pos), int(cat_horz_pos)), (int(mouse_vert_pos), int(mouse_horz_pos))


def action_to_action_index(action):
    # note that action X means the agent remains where it is
    valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    if not action in valid_actions:
        raise Exception('Invalid action')
    return int(valid_actions.index(action))


def action_index_to_action(action_index):
    if not action_index in range(9):
        raise Exception('action_index must be between 0 and 9')
    valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    return valid_actions[action_index]


def action_to_moves(action):
    # takes action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
    # returns vert_move, horz_move
    valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    if not action in valid_actions:
        raise Exception('Invalid action')
    action_index = valid_actions.index(action)
    horz_move = float(np.mod(action_index, 3) - 1)
    vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
    return int(vert_move), int(horz_move)


def moves_to_action(vert_move, horz_move):
    # takes vert_move, horz_move
    # returns action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
    if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
        raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
    valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    return valid_actions[(vert_move + 1 ) * 3 + (horz_move + 1)]


def action_index_to_moves(action_index):
    if not action_index in range(9):
        raise Exception('action_index must be between 0 and 9')
    horz_move = float(np.mod(action_index, 3) - 1)
    vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
    return int(vert_move), int(horz_move)


def moves_to_action_index(vert_move, horz_move):
    if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
        raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
    return int((vert_move + 1 ) * 3 + (horz_move + 1))


def move_is_legal(pos, vert_move, horz_move, board_height, board_width, walls = None):
    if not (((pos[0] + vert_move) in range(board_height)) and ((pos[1] + horz_move) in range(board_width))):
        # attempted move would go outside board
        return False
    if not walls == None:
        if (vert_move != 0 and horz_move != 0):
            vertex_point = (pos[0] + 0.5 * vert_move, pos[1] + 0.5 * horz_move)
            for wall in walls:
                if (abs(wall[0] - vertex_point[0]) <= 0.5 and abs(wall[1] - vertex_point[1]) <= 0.5):
                    return False
            return True
        elif (pos[0] + 0.5 * vert_move, pos[1] + 0.5 * horz_move) in walls:
            return False
    return True


# def move_stays_on_board(pos, vert_move, horz_move, board_height, board_width):
#     return ((pos[0] + vert_move) in range(board_height)) and ((pos[1] + horz_move) in range(board_width))


def extract_training_metadata(filename):
    if filename[-4:] == '.pth':
        filename = filename[:-4]
        start_index_of_parent_dir = filename.find('dqn_weights')
        if not start_index_of_parent_dir == -1:
            filename = filename[start_index_of_parent_dir + len('dqn_weights/'):]
        start_index_of_parent_dir = filename.find('drqn_weights')
        if not start_index_of_parent_dir == -1:
            filename = filename[start_index_of_parent_dir + len('drqn_weights/'):]
        metadata_list = filename.split('_')
        metadata_dict = {}
        metadata_dict['board_height'] = metadata_list[2]
        metadata_dict['board_width'] = metadata_list[3]
        metadata_dict['reward_type'] = metadata_list[4]
        metadata_dict['no_episodes'] = metadata_list[5]
        # metadata_dict['max_t'] = metadata_list[6]
        metadata_dict['eps_start'] = metadata_list[6]
        metadata_dict['eps_end'] = metadata_list[7]
        metadata_dict['eps_decay'] = metadata_list[8]
        metadata_dict['sight'] = metadata_list[9]
        metadata_dict['use_belief_state'] = metadata_list[10]
        metadata_dict['algorithm'] = metadata_list[11]
        return metadata_dict
    elif filename[-4:] == '.txt':
        filename = filename[:-4]
        start_index_of_parent_dir = filename.find('qlearning_qvalues')
        if not start_index_of_parent_dir == -1:
            filename = filename[start_index_of_parent_dir + len('qlearning_qvalues/'):]
        metadata_list = filename.split('_')
        metadata_dict = {}
        metadata_dict['board_height'] = metadata_list[2]
        metadata_dict['board_width'] = metadata_list[3]
        metadata_dict['reward_type'] = metadata_list[4]
        metadata_dict['no_episodes'] = metadata_list[5]
        metadata_dict['discount_factor'] = metadata_list[6]
        metadata_dict['alpha'] = metadata_list[7]

        # metadata_dict['epsilon'] = metadata_list_2[8]
        metadata_dict['eps_start'] = metadata_list[8]
        metadata_dict['eps_end'] = metadata_list[9]
        metadata_dict['eps_decay'] = metadata_list[10]

        metadata_dict['sight'] = metadata_list[11]
        return metadata_dict
    else:
        raise Exception('Unrecognised file type')


def extract_analysis_metadata(filepath):
    if filepath.find('/') == -1:
        raise Exception('Filepath must include parent directory, not just filename')
    start_index_of_parent_dir = filepath.find('training_analysis')
    if not start_index_of_parent_dir == -1:
        filepath = filepath[start_index_of_parent_dir + len('training_analysis/'):]
    metadata_list_1 = filepath.split('/')
    metadata_list_2 = metadata_list_1[1].split('_')
    if metadata_list_1[0] == 'dqn':
        metadata_dict = {}
        metadata_dict['training_algorithm'] = 'dqn'
        metadata_dict['board_height'] = metadata_list_2[2]
        metadata_dict['board_width'] = metadata_list_2[3]
        metadata_dict['reward_type'] = metadata_list_2[4]
        metadata_dict['no_episodes'] = metadata_list_2[5]
        # metadata_dict['max_t'] = metadata_list[6]
        metadata_dict['eps_start'] = metadata_list_2[6]
        metadata_dict['eps_end'] = metadata_list_2[7]
        metadata_dict['eps_decay'] = metadata_list_2[8]
        metadata_dict['sight'] = metadata_list_2[9]
        metadata_dict['use_belief_state'] = metadata_list_2[10]
        return metadata_dict
    elif metadata_list_1[0] == 'qlearning':
        metadata_dict = {}
        metadata_dict['training_algorithm'] = 'qlearning'
        metadata_dict['board_height'] = metadata_list_2[2]
        metadata_dict['board_width'] = metadata_list_2[3]
        metadata_dict['reward_type'] = metadata_list_2[4]
        metadata_dict['no_episodes'] = metadata_list_2[5]
        metadata_dict['discount_factor'] = metadata_list_2[6]
        metadata_dict['alpha'] = metadata_list_2[7]

        # metadata_dict['epsilon'] = metadata_list_2[8]
        metadata_dict['eps_start'] = metadata_list_2[8]
        metadata_dict['eps_end'] = metadata_list_2[9]
        metadata_dict['eps_decay'] = metadata_list_2[10]

        metadata_dict['sight'] = metadata_list_2[11]
        return metadata_dict
    else:
        raise Exception('Unrecognised filepath - remember to include the parent directory')


def observed_state_as_qlearning_state(board_height, board_width, true_state_index, sight, walls):
    cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)
    if cat_can_see_mouse(cat_pos, mouse_pos, sight, walls):
        return true_state_index
    else:
        new_state_vector = [true_state_index[i] * (i<board_height*board_width) for i in range(len(true_state_index))]
        new_state_vector = true_


def observed_state_as_nn_input(board_height, board_width, true_state_index, previous_observed_state_nn_input, sight, use_belief_state, walls):
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

    if (sight == None or cat_can_see_mouse(true_cat_pos, true_mouse_pos, sight, walls = walls)):
        mouse_pos_dist[true_mouse_pos_index] = 1
    elif use_belief_state == False:
        total_unseen_positions = 0
        for row in range(board_height):
            for col in range(board_width):
                mouse_board_pos = (row, col)
                mouse_board_pos_index = board_pos_to_board_pos_index(mouse_board_pos, board_height, board_width)
                # board_pos_index = board_pos_to_board_pos_index(board_pos, board_height, board_width)
                if not cat_can_see_mouse(true_cat_pos, mouse_board_pos, sight, walls = walls):
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
                        mouse_move_stays_on_board = move_is_legal(previous_mouse_pos, mouse_vert_move, mouse_horz_move, board_height, board_width, walls = walls)
                        new_mouse_vert_pos = previous_mouse_row + mouse_vert_move * mouse_move_stays_on_board
                        new_mouse_horz_pos = previous_mouse_col + mouse_horz_move * mouse_move_stays_on_board
                        new_mouse_pos = (new_mouse_vert_pos, new_mouse_horz_pos)
                        new_mouse_pos_index = board_pos_to_board_pos_index(new_mouse_pos, board_height, board_width)
                        if not cat_can_see_mouse(true_cat_pos, new_mouse_pos, sight, walls = walls):
                            mouse_pos_dist[new_mouse_pos_index] += (1/9) * previous_mouse_pos_dist[previous_mouse_pos_index]
        mouse_pos_dist = mouse_pos_dist/sum(mouse_pos_dist)

    # cat_output = np.array([list(np.zeros(board_width)) for row in range(board_height)])
    # for index, prob in enumerate(cat_pos_one_hot):
    #     row, col = board_pos_index_to_board_pos(index, board_height, board_width)
    #     cat_output[row][col] = prob
    # print(cat_output)
    #
    # mouse_output = np.array([list(np.zeros(board_width)) for row in range(board_height)])
    # for index, prob in enumerate(mouse_pos_dist):
    #     row, col = board_pos_index_to_board_pos(index, board_height, board_width)
    #     mouse_output[row][col] = prob
    # print(mouse_output)

    return np.concatenate((cat_pos_one_hot, mouse_pos_dist))


def nn_input_as_drqn_input(board_height, board_width, nn_input):
    output = torch.zeros(1, 1, 2 * board_height * board_width)
    output[0][0] = torch.from_numpy(nn_input)
    return output

def save_policy_to_file(policy, filename):
    with open(filename, 'w') as file:
        for key, value in policy.items():
            file.write('%s:%s\n' % (key, value))
        file.close()


def load_policy_from_file(filename):
    policy = {}
    with open(filename) as file:
        for item in file:
            if ':' in item:
                key,value = item.split(':', 1)
                key = int(key.rstrip())
                value = int(value.rstrip())
                policy[key]=value
            else:
                pass # deal with bad lines of text here
    return policy


def save_qvalues_to_file(qvalues, filename):
    with open(filename, 'w') as file:
        for key, value in qvalues.items():
            qvalues = ",".join([str(elt) for elt in value.values()])
            file.write('%s:%s\n' % (key, qvalues))
        file.close()


def load_qvalues_from_file(filename):
    qvalues = {}
    with open(filename) as file:
        for item in file:
            if ':' in item:
                key,value = item.split(':', 1)
                key = int(key.rstrip())
                qvalues_list = value.split(',')
                qvalues_dict = {}
                for action, qvalue in enumerate(qvalues_list):
                    qvalues_dict[action] = float(qvalue.rstrip())
                qvalues[key] = qvalues_dict
            else:
                pass # deal with bad lines of text here
    return qvalues


def save_training_analysis_to_file(stats, filename):
    with open(filename, 'w') as file:
        file.write(','.join(np.array(stats.saved_episodes).astype(str)))
        file.write('\n')
        file.write(','.join(stats.episode_lengths.astype(str)))
        file.write('\n')
        file.write(','.join(stats.episode_rewards.astype(str)))
        file.close()


def load_training_analysis_from_file(filepath):
    filename = 'stats.txt'
    if not filepath[-9:] == filename:
        filepath = filepath + '/' + filename
    with open(filepath) as file:
        lines = file.readlines()
        saved_episodes = lines[0].split(',')
        episode_lengths = lines[1].split(',')
        episode_rewards = lines[2].split(',')
        file.close()
    stats = namedtuple("Stats",["saved_episodes", "episode_lengths", "episode_rewards"])(
        saved_episodes = [float(item) for item in saved_episodes],
        episode_lengths=[float(item) for item in episode_lengths],
        episode_rewards=[float(item) for item in episode_rewards])
    return stats


def print_q_qlearning(q_values_filename, state_index):
    q_values_dict = load_qvalues_from_file(q_values_filename)
    print('state: ', state_index, '\n q values: ', q_values_dict[state_index])

def print_q_dqn(parameter_filename, state_index, board_height, board_width, device = None):

    metadata = extract_training_metadata(parameter_filename)
    if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
        raise Exception('Parameters were generated using different board size')
    if parameter_filename.find('dqn_weights/') == -1:
        parameter_filename = 'dqn_weights/' + parameter_filename
        if parameter_filename.find('trained_parameters/') == -1:
            parameter_filename = 'trained_parameters/' + parameter_filename
    dqn_agent = DQNAgent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
    dqn_agent.qnetwork_behaviour.load_state_dict(torch.load(parameter_filename))

    cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)
    state_nn_input = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
    # observed_state_as_nn_input(board_height, board_width, state_index, initial_state_distribution, sight = sight, use_belief_state = use_belief_state)
    state_tensor = torch.from_numpy(state_nn_input).float().unsqueeze(0).to(device)
    dqn_agent.qnetwork_behaviour.eval()
    with torch.no_grad():
        action_values = dqn_agent.qnetwork_behaviour(state_tensor)
    dqn_agent.qnetwork_behaviour.train()

    print('state: ', state_index, '\n q values: ', action_values)



    # for cat_vert_pos in range(board_height):
    #     for cat_horz_pos in range(board_width):
    #         nn_state = positions_to_nn_input((cat_vert_pos, cat_horz_pos), mouse_pos, board_height, board_width)
    #         cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = agent.act(nn_state)
    #
    #
    #
    #
    #
    # cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)
    # state_nn_input = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
    # # observed_state_as_nn_input(board_height, board_width, state_index, initial_state_distribution, sight = sight, use_belief_state = use_belief_state)
    # state_tensor = torch.from_numpy(state_nn_input).float().unsqueeze(0).to(device)
    # agent.qnetwork_behaviour.eval()
    # with torch.no_grad():
    #     action_values = agent.qnetwork_behaviour(state_tensor)
    # agent.qnetwork_behaviour.train()


def print_q(parameter_filename, state_index, board_height, board_width, device = None):

    if type(parameter_filename) is str and parameter_filename[-4:] == '.txt':
        metadata = extract_training_metadata(parameter_filename)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Parameters were generated using different board size')
        if parameter_filename.find('qlearning_qvalues/') == -1:
            parameter_filename = 'qlearning_qvalues/' + parameter_filename
            if parameter_filename.find('trained_parameters/') == -1:
                parameter_filename = 'trained_parameters/' + parameter_filename
        q_values_dict = load_qvalues_from_file(parameter_filename)
        print('state: ', state_index, '\n q values: ', q_values_dict[state_index])
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.pth':

        metadata = extract_training_metadata(parameter_filename)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Parameters were generated using different board size')
        if parameter_filename.find('dqn_weights/') == -1:
            parameter_filename = 'dqn_weights/' + parameter_filename
            if parameter_filename.find('trained_parameters/') == -1:
                parameter_filename = 'trained_parameters/' + parameter_filename
        dqn_agent = DQNAgent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
        dqn_agent.qnetwork_behaviour.load_state_dict(torch.load(parameter_filename))

        cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)
        state_nn_input = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
        # observed_state_as_nn_input(board_height, board_width, state_index, initial_state_distribution, sight = sight, use_belief_state = use_belief_state)
        state_tensor = torch.from_numpy(state_nn_input).float().unsqueeze(0).to(device)
        dqn_agent.qnetwork_behaviour.eval()
        with torch.no_grad():
            action_values = dqn_agent.qnetwork_behaviour(state_tensor)
        dqn_agent.qnetwork_behaviour.train()

        print('state: ', state_index, '\n q values: ', action_values)
    else:
        raise ValueError('parameter_filename type not recognised. Should be .txt or .pth filename')


def wall_midpoint_to_coords(wall_midpoint):
    wall_midpoint_row, wall_midpoint_col = wall_midpoint
    x_coord = wall_midpoint_col + 0.5
    y_coord = wall_midpoint_row + 0.5
    return (x_coord, y_coord)
