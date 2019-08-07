import numpy as np


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


def initialise_mouse_prob_dist(height, width, cat_pos, mouse_pos, sight):
    if cat_can_see_mouse(cat_pos, mouse_pos, sight):
        mouse_pos_prob_dist = np.zeros((height, width))
        mouse_pos_prob_dist[mouse_pos] = 1
    else:
        mouse_pos_prob_dist = np.ones((height, width))
        no_observable_cells = 0
        for row_offset in range(-sight,sight+1):
            for col_offset in range(-sight,sight+1):
                if ((cat_pos[0] + row_offset in range(height)) and (cat_pos[1] + col_offset in range(width))):
                    no_observable_cells += 1
                    mouse_pos_prob_dist[cat_pos[0] + row_offset, cat_pos[1] + col_offset] = 0
        mouse_pos_prob_dist *= 1/(height * width - no_observable_cells)
    return mouse_pos_prob_dist


def cat_can_see_mouse(cat_pos, mouse_pos, sight):
    # assumimg cat has can see sight squares in each direction (i.e. can see a (1+2xsight)x(1+2xsight) grid around itself)
    return ((abs(cat_pos[0]-mouse_pos[0]) <= sight) and (abs(cat_pos[1]-mouse_pos[1]) <= sight))


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


def move_stays_on_board(pos, vert_move, horz_move, board_height, board_width):
    return ((pos[0] + vert_move) in range(board_height)) and ((pos[1] + horz_move) in range(board_width))


def stabilisation_analysis(stats, averaging_window = 1000, mean_tolerance = 5, var_tolerance = 25):
    i = 0
    while i + 2 * averaging_window - 1 <= len(stats[0]):
        average_window_1 = np.mean(stats.episode_lengths[i:i + averaging_window - 1])
        # average_window_2 = np.mean(stats.episode_lengths[i + averaging_window:i + 2 * averaging_window - 1])
        average_window_2 = np.mean(stats.episode_lengths[i + averaging_window:])
        variance_whole_window = np.var(stats.episode_lengths[i:i + 2 * averaging_window - 1])
        if ((abs(average_window_1 - average_window_2) < mean_tolerance) \
        and variance_whole_window < var_tolerance):
            return i, round(np.mean(stats.episode_lengths[i:]),2)
        else:
            i += averaging_window
    raise Exception('Insufficient episodes for episode lengths to stabilise')

def extract_training_metadata(filename):
    if filename[-4:] == '.pth':
        filename = filename[:-4]
        start_index_of_parent_dir = filename.find('dqn_weights')
        if not start_index_of_parent_dir == -1:
            filename = filename[start_index_of_parent_dir + len('dqn_weights/'):]
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
        return metadata_dict
    elif filename[-4:] == '.txt':
        filename = filename[:-4]
        start_index_of_parent_dir = filename.find('qlearning_policies')
        if not start_index_of_parent_dir == -1:
            filename = filename[start_index_of_parent_dir + len('qlearning_policies/'):]
        metadata_list = filename.split('_')
        metadata_dict = {}
        metadata_dict['board_height'] = metadata_list[2]
        metadata_dict['board_width'] = metadata_list[3]
        metadata_dict['reward_type'] = metadata_list[4]
        metadata_dict['no_episodes'] = metadata_list[5]
        metadata_dict['discount_factor'] = metadata_list[6]
        metadata_dict['alpha'] = metadata_list[7]
        metadata_dict['epsilon'] = metadata_list[8]
        metadata_dict['sight'] = metadata_list[9]
        return metadata_dict
    else:
        raise Exception('Unrecognised file type')


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
