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


def initialise_mouse_prob_dist(height, width, cat_pos, mouse_pos):
    if cat_can_see_mouse(cat_pos, mouse_pos):
        mouse_pos_prob_dist = np.zeros((height, width))
        mouse_pos_prob_dist[mouse_pos] = 1
    else:
        mouse_pos_prob_dist = np.ones((height, width))
        no_observable_cells = 0
        for row_offset in [-2, -1, 0, 1, 2]:
            for col_offset in [-2, -1, 0, 1, 2]:
                if ((cat_pos[0] + row_offset in range(height)) and (cat_pos[1] + col_offset in range(width))):
                    no_observable_cells += 1
                    mouse_pos_prob_dist[cat_pos[0] + row_offset, cat_pos[1] + col_offset] = 0
        mouse_pos_prob_dist *= 1/(height * width - no_observable_cells)
    return mouse_pos_prob_dist

def cat_can_see_mouse(cat_pos, mouse_pos):
    # assumimg cat has sight 2 in each direction (i.e. can see a 5x5 grid around iteself)
    return ((abs(cat_pos[0]-mouse_pos[0])<=2) and (abs(cat_pos[1]-mouse_pos[1])<=2))


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
