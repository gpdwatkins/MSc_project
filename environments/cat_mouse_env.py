import numpy as np
import sys
from gym.envs.toy_text import discrete
from lib.utils import *

actions = ['NW','N','NE','W','X','E','SW','S','SE']


def categorical_sample(prob_n, np_random = None):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np.random.rand()).argmax()


class CatMouseEnv_binary_reward(discrete.DiscreteEnv):
    """
    An implementation of the pursuer-evader (cat-and-mouse) reinforcement
    learning task.

    Adapted from code provided on applenob's Github https://github.com/applenob
    and Javen Chen's blog https://applenob.github.io/cliff_walking.html

    (In turn this was adapted from Example 6.6 (page 106) from
    Reinforcement Learning: An Introduction by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)

    The board is a matrix with dimensions specified when the environment is initialised.

    In the binary_reward environment, a reward of 1 is received when the cat catches
    the mouse (and this terminates the episode). All other moves incur 0 reward.
    """
    metadata = {'render.modes': ['human', 'ansi']}


    def __init__(self, board_height, board_width):
        self.board_height = board_height
        self.board_width = board_width
        self.reward_type = 'bin'

        nS = self.board_height * self.board_width * self.board_height * self.board_width
        nA = len(actions)

        P = {}
        for state_index in range(nS):
            (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(state_index, self.board_height, self.board_width)
            # state_index = self._positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos))
            P[state_index] = {}
            for action in actions:
                cat_vert_move, cat_horz_move = action_to_moves(action)
                cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(self.board_height)) and ((cat_horz_pos + cat_horz_move) in range(self.board_width))
                new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
                new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
                new_state_instances = {}
                mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
                for mouse_vert_move in mouse_vert_moves:
                    for mouse_horz_move in mouse_horz_moves:
                        # mouse_action = self._moves_to_action(mouse_vert_move, mouse_horz_move)
                        mouse_move_stays_on_board = move_stays_on_board((mouse_vert_pos, mouse_horz_pos), mouse_vert_move, mouse_horz_move, self.board_height, self.board_width)
                        new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
                        new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
                        new_state = positions_to_state_index((new_cat_vert_pos, new_cat_horz_pos), (new_mouse_vert_pos, new_mouse_horz_pos), self.board_height, self.board_width)
                        if not new_state in new_state_instances.keys():
                            game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
                            new_state_instances[new_state] = [1, game_over]
                        else:
                            new_state_instances[new_state][0] += 1
                possible_next_states_list = []
                for new_state, (no_instances, game_over) in new_state_instances.items():
                    possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, 1 * game_over, game_over)
                    possible_next_states_list.append(possible_next_state_tuple)
                P[state_index][action_to_action_index(action)] = possible_next_states_list

        # Calculate initial state distribution
        isd = np.ones(nS)
        for state_index in range(nS):
            (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(state_index, self.board_height, self.board_width)
            if ((cat_vert_pos == mouse_vert_pos) and (cat_horz_pos == mouse_horz_pos)):
                isd[state_index] = 0
        isd = isd/sum(isd)

        super(CatMouseEnv_binary_reward, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = sys.stdout

        for row in range(self.board_height):
            for col in range(self.board_width):
                (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(self.s, self.board_height, self.board_width)
                if (cat_vert_pos == row and cat_horz_pos == col and mouse_vert_pos == row and mouse_horz_pos == col):
                    output = " @ "
                elif (cat_vert_pos == row and cat_horz_pos == col):
                    output = " C "
                elif (mouse_vert_pos == row and mouse_horz_pos == col):
                    output = " M "
                else:
                    output = " o "
                if col == 0:
                    output = output.lstrip()
                if col == (self.board_width - 1):
                    output = output.rstrip()
                    output += '\n'
                outfile.write(output)
        outfile.write('\n')


    # def _state_index_to_positions(self, state_index):
    #     if not state_index in range(self.board_height**2 * self.board_width**2):
    #         raise ValueError('state_index must be between 0 and %d' % (self.board_height**2 * self.board_width**2 - 1))
    #
    #     cat_vert_pos = float(np.mod(state_index,self.board_height))
    #     cat_horz_pos = np.mod(state_index - cat_vert_pos, self.board_height * self.board_width) / self.board_height
    #     mouse_vert_pos = np.mod(state_index - cat_horz_pos * self.board_height - cat_vert_pos, self.board_height * self.board_width * self.board_height) / (self.board_height * self.board_width)
    #     mouse_horz_pos = (state_index - mouse_vert_pos * self.board_height * self.board_width - cat_horz_pos * self.board_height - cat_vert_pos) / (self.board_height * self.board_width * self.board_height)
    #     return (int(cat_vert_pos), int(cat_horz_pos)), (int(mouse_vert_pos), int(mouse_horz_pos))
    #
    #
    # def _positions_to_state_index(self, cat_pos, mouse_pos):
    #     if not np.prod([ \
    #     cat_pos[0] in range(self.board_height), \
    #     cat_pos[1] in range(self.board_width), \
    #     ]):
    #         raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))
    #
    #     if not np.prod([ \
    #     mouse_pos[0] in range(self.board_height), \
    #     mouse_pos[1] in range(self.board_width) \
    #     ]):
    #         raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))
    #
    #     return int( \
    #     cat_pos[0] + \
    #     cat_pos[1] * self.board_height  + \
    #     mouse_pos[0] * self.board_height * self.board_width + \
    #     mouse_pos[1] * self.board_height * self.board_width * self.board_height \
    #     )
    #
    #
    # def _action_to_action_index(self, action):
    #     # note that action X means the agent remains where it is
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     if not action in valid_actions:
    #         raise Exception('Invalid action')
    #     return int(valid_actions.index(action))
    #
    #
    # def _action_index_to_action(self, action_index):
    #     if not action_index in range(9):
    #         raise Exception('action_index must be between 0 and 9')
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     return valid_actions[action_index]
    #
    #
    # def _action_to_moves(self, action):
    #     # takes action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
    #     # returns vert_move, horz_move
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     if not action in valid_actions:
    #         raise Exception('Invalid action')
    #     action_index = valid_actions.index(action)
    #     horz_move = float(np.mod(action_index, 3) - 1)
    #     vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
    #     return int(vert_move), int(horz_move)
    #
    #
    # def _moves_to_action(self, vert_move, horz_move):
    #     # takes vert_move, horz_move
    #     # returns action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
    #     if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
    #         raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     return valid_actions[(vert_move + 1 ) * 3 + (horz_move + 1)]
    #
    #
    # def _action_index_to_moves(self, action_index):
    #     if not action_index in range(9):
    #         raise Exception('action_index must be between 0 and 9')
    #     horz_move = float(np.mod(action_index, 3) - 1)
    #     vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
    #     return int(vert_move), int(horz_move)
    #
    #
    # def _moves_to_action_index(vert_move, horz_move):
    #     if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
    #         raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
    #     return int((vert_move + 1 ) * 3 + (horz_move + 1))


class CatMouseEnv_proximity_reward(discrete.DiscreteEnv):
    """
    An implementation of the pursuer-evader (cat-and-mouse)
    reinforcement learning task.

    Adapted from code provided on applenob's Github https://github.com/applenob
    and Javen Chen's blog https://applenob.github.io/cliff_walking.html

    (In turn this was adapted from Example 6.6 (page 106) from
    Reinforcement Learning: An Introduction by Sutton and Barto:
    http://incompleteideas.net/book/bookdraft2018jan1.pdf

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)

    The board is a matrix with dimensions specified when the environment is initialised.

    In the proximity_reward environment, a reward of board_height * board_width is received
    when the cat catches the mouse (and this terminates the episode). All other moves incur
    reward equal to the distance between the cat and the mouse (defined to be the min
    number of moves it could take for the cat to catch the mouse if the mouse didn't move).
    """
    metadata = {'render.modes': ['human', 'ansi']}


    def __init__(self, board_height, board_width):
        self.board_height = board_height
        self.board_width = board_width
        self.reward_type = 'prox'

        nS = self.board_height * self.board_width * self.board_height * self.board_width
        nA = len(actions)

        P = {}
        for state_index in range(nS):
            (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(state_index, self.board_height, self.board_width)
            # state_index = self._positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos))
            P[state_index] = {}
            for action in actions:
                cat_vert_move, cat_horz_move = action_to_moves(action)
                cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(self.board_height)) and ((cat_horz_pos + cat_horz_move) in range(self.board_width))
                new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
                new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
                new_state_instances = {}
                mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
                for mouse_vert_move in mouse_vert_moves:
                    for mouse_horz_move in mouse_horz_moves:
                        # mouse_action = self._moves_to_action(mouse_vert_move, mouse_horz_move)
                        mouse_move_stays_on_board = move_stays_on_board((mouse_vert_pos, mouse_horz_pos), mouse_vert_move, mouse_horz_move, self.board_height, self.board_width)
                        new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
                        new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
                        new_state = positions_to_state_index((new_cat_vert_pos, new_cat_horz_pos), (new_mouse_vert_pos, new_mouse_horz_pos), self.board_height, self.board_width)
                        if not new_state in new_state_instances.keys():
                            game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
                            new_state_instances[new_state] = [1, game_over, new_cat_vert_pos, new_cat_horz_pos, new_mouse_vert_pos, new_mouse_horz_pos]
                        else:
                            new_state_instances[new_state][0] += 1
                possible_next_states_list = []
                for new_state, (no_instances, game_over, new_cat_vert_pos, new_cat_horz_pos, new_mouse_vert_pos, new_mouse_horz_pos) in new_state_instances.items():
                    if game_over:
                        reward = self.board_height * self.board_width
                    else:
                        cat_mouse_separation = max(abs(new_cat_vert_pos - new_mouse_vert_pos), abs(new_cat_horz_pos - new_mouse_horz_pos))
                        reward = 0 - 2 * cat_mouse_separation
                    possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, int(reward), game_over)
                    possible_next_states_list.append(possible_next_state_tuple)
                P[state_index][action_to_action_index(action)] = possible_next_states_list

        isd = np.ones(nS)
        for state_index in range(nS):
            (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(state_index, self.board_height, self.board_width)
            if ((cat_vert_pos == mouse_vert_pos) and (cat_horz_pos == mouse_horz_pos)):
                isd[state_index] = 0
        isd = isd/sum(isd)

        super(CatMouseEnv_proximity_reward, self).__init__(nS, nA, P, isd)


    def render(self, mode='human'):
        outfile = sys.stdout

        for row in range(self.board_height):
            for col in range(self.board_width):
                (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = state_index_to_positions(self.s, self.board_height, self.board_width)
                if (cat_vert_pos == row and cat_horz_pos == col and mouse_vert_pos == row and mouse_horz_pos == col):
                    output = " @ "
                elif (cat_vert_pos == row and cat_horz_pos == col):
                    output = " C "
                elif (mouse_vert_pos == row and mouse_horz_pos == col):
                    output = " M "
                else:
                    output = " o "
                if col == 0:
                    output = output.lstrip()
                if col == (self.board_width - 1):
                    output = output.rstrip()
                    output += '\n'
                outfile.write(output)
        outfile.write('\n')


    # def _state_index_to_positions(self, state_index):
    #     if not state_index in range(self.board_height**2 * self.board_width**2):
    #         raise ValueError('state_index must be between 0 and %d' % (self.board_height**2 * self.board_width**2 - 1))
    #
    #     cat_vert_pos = float(np.mod(state_index,self.board_height))
    #     cat_horz_pos = np.mod(state_index - cat_vert_pos, self.board_height * self.board_width) / self.board_height
    #     mouse_vert_pos = np.mod(state_index - cat_horz_pos * self.board_height - cat_vert_pos, self.board_height * self.board_width * self.board_height) / (self.board_height * self.board_width)
    #     mouse_horz_pos = (state_index - mouse_vert_pos * self.board_height * self.board_width - cat_horz_pos * self.board_height - cat_vert_pos) / (self.board_height * self.board_width * self.board_height)
    #     return (int(cat_vert_pos), int(cat_horz_pos)), (int(mouse_vert_pos), int(mouse_horz_pos))
    #
    #
    # def _positions_to_state_index(self, cat_pos, mouse_pos):
    #     if not np.prod([ \
    #     cat_pos[0] in range(self.board_height), \
    #     cat_pos[1] in range(self.board_width), \
    #     ]):
    #         raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))
    #
    #     if not np.prod([ \
    #     mouse_pos[0] in range(self.board_height), \
    #     mouse_pos[1] in range(self.board_width) \
    #     ]):
    #         raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))
    #
    #     return int( \
    #     cat_pos[0] + \
    #     cat_pos[1] * self.board_height  + \
    #     mouse_pos[0] * self.board_height * self.board_width + \
    #     mouse_pos[1] * self.board_height * self.board_width * self.board_height \
    #     )
    #
    #
    # def _action_to_action_index(self, action):
    #     # note that action X means the agent remains where it is
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     if not action in valid_actions:
    #         raise Exception('Invalid action')
    #     return int(valid_actions.index(action))
    #
    #
    # def _action_index_to_action(self, action_index):
    #     if not action_index in range(9):
    #         raise Exception('action_index must be between 0 and 9')
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     return valid_actions[action_index]
    #
    #
    # def _action_to_moves(self, action):
    #     # takes action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
    #     # returns vert_move, horz_move
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     if not action in valid_actions:
    #         raise Exception('Invalid action')
    #     action_index = valid_actions.index(action)
    #     horz_move = float(np.mod(action_index, 3) - 1)
    #     vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
    #     return int(vert_move), int(horz_move)
    #
    #
    # def _moves_to_action(self, vert_move, horz_move):
    #     # takes vert_move, horz_move
    #     # returns action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
    #     if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
    #         raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
    #     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
    #     return valid_actions[(vert_move + 1 ) * 3 + (horz_move + 1)]
    #
    #
    # def _action_index_to_moves(self, action_index):
    #     if not action_index in range(9):
    #         raise Exception('action_index must be between 0 and 9')
    #     horz_move = float(np.mod(action_index, 3) - 1)
    #     vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
    #     return int(vert_move), int(horz_move)
    #
    #
    # def _moves_to_action_index(vert_move, horz_move):
    #     if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
    #         raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
    #     return int((vert_move + 1 ) * 3 + (horz_move + 1))


# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv new versions using different spaces vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# from gym.envs.toy_text import discrete
#
# class CatMouseEnv_proximity_reward_box(discrete.DiscreteEnv):
#     """
#     An implementation of the pursuer-evader (cat-and-mouse)
#     reinforcement learning task.
#
#     Adapted from code provided on applenob's Github https://github.com/applenob
#     and Javen Chen's blog https://applenob.github.io/cliff_walking.html
#
#     (In turn this was adapted from Example 6.6 (page 106) from
#     Reinforcement Learning: An Introduction by Sutton and Barto:
#     http://incompleteideas.net/book/bookdraft2018jan1.pdf
#
#     With inspiration from:
#     https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py)
#
#     The board is a matrix with dimensions specified when the environment is initialised.
#
#     In the proximity_reward environment, a reward of board_height * board_width is received
#     when the cat catches the mouse (and this terminates the episode). All other moves incur
#     reward equal to the distance between the cat and the mouse (defined to be the min
#     number of moves it could take for the cat to catch the mouse if the mouse didn't move).
#
#     Note that everything in this class descibes exactly the environment. If the agent does
#     not have perfect information (i.e. there is partial obervability), it may only 'know'
#     a subset of the current state.
#     """
#     metadata = {'render.modes': ['human', 'ansi']}
#
#
#     def __init__(self, board_height, board_width):
#         self.board_height = board_height
#         self.board_width = board_width
#         self.reward_type = 'prox'
#
#         nS = self.board_height * self.board_width * self.board_height * self.board_width
#         nA = len(actions)
#
#         P = {}
#         for state_index in range(nS):
#             (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = self._state_index_to_positions(s)
#             # state_index = self._positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos))
#             P[state_index] = {}
#             for action in actions:
#                 cat_vert_move, cat_horz_move = self._action_to_moves(action)
#                 cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(self.board_height)) and ((cat_horz_pos + cat_horz_move) in range(self.board_width))
#                 new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
#                 new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
#                 new_state_instances = {}
#                 mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
#                 for mouse_vert_move in mouse_vert_moves:
#                     for mouse_horz_move in mouse_horz_moves:
#                         # mouse_action = self._moves_to_action(mouse_vert_move, mouse_horz_move)
#                         mouse_move_stays_on_board = move_stays_on_board((mouse_vert_pos, mouse_horz_pos), mouse_vert_move, mouse_horz_move, self.board_height, self.board_width)
#                         new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
#                         new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
#                         new_state = self._positions_to_state_index((new_cat_vert_pos, new_cat_horz_pos), (new_mouse_vert_pos, new_mouse_horz_pos))
#                         if not new_state in new_state_instances.keys():
#                             game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
#                             new_state_instances[new_state] = [1, game_over, new_cat_vert_pos, new_cat_horz_pos, new_mouse_vert_pos, new_mouse_horz_pos]
#                         else:
#                             new_state_instances[new_state][0] += 1
#                 possible_next_states_list = []
#                 for new_state, (no_instances, game_over, new_cat_vert_pos, new_cat_horz_pos, new_mouse_vert_pos, new_mouse_horz_pos) in new_state_instances.items():
#                     if game_over:
#                         reward = self.board_height * self.board_width
#                     else:
#                         cat_mouse_separation = max(abs(new_cat_vert_pos - new_mouse_vert_pos), abs(new_cat_horz_pos - new_mouse_horz_pos))
#                         reward = 0 - cat_mouse_separation
#                     possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, int(reward), game_over)
#                     possible_next_states_list.append(possible_next_state_tuple)
#                 P[state_index][self._action_to_action_index(action)] = possible_next_states_list
#
#         isd = np.ones(nS)
#         for state_index in range(nS):
#             (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = self._state_index_to_positions(state_index)
#             if ((cat_vert_pos == mouse_vert_pos) and (cat_horz_pos == mouse_horz_pos)):
#                 isd[state_index] = 0
#         isd = isd/sum(isd)
#
#         super(CatMouseEnv_proximity_reward, self).__init__(nS, nA, P, isd)
#
#
#     def render(self, mode='human'):
#         outfile = sys.stdout
#
#         for row in range(self.board_height):
#             for col in range(self.board_width):
#                 (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = self._state_index_to_positions(self.s)
#                 if (cat_vert_pos == row and cat_horz_pos == col and mouse_vert_pos == row and mouse_horz_pos == col):
#                     output = " @ "
#                 elif (cat_vert_pos == row and cat_horz_pos == col):
#                     output = " C "
#                 elif (mouse_vert_pos == row and mouse_horz_pos == col):
#                     output = " M "
#                 else:
#                     output = " o "
#                 if col == 0:
#                     output = output.lstrip()
#                 if col == (self.board_width - 1):
#                     output = output.rstrip()
#                     output += '\n'
#                 outfile.write(output)
#         outfile.write('\n')
#
#
#     def _state_index_to_positions(self, state_index):
#         if not state_index in range(self.board_height**2 * self.board_width**2):
#             raise ValueError('state_index must be between 0 and %d' % (self.board_height**2 * self.board_width**2 - 1))
#
#         cat_vert_pos = float(np.mod(state_index,self.board_height))
#         cat_horz_pos = np.mod(state_index - cat_vert_pos, self.board_height * self.board_width) / self.board_height
#         mouse_vert_pos = np.mod(state_index - cat_horz_pos * self.board_height - cat_vert_pos, self.board_height * self.board_width * self.board_height) / (self.board_height * self.board_width)
#         mouse_horz_pos = (state_index - mouse_vert_pos * self.board_height * self.board_width - cat_horz_pos * self.board_height - cat_vert_pos) / (self.board_height * self.board_width * self.board_height)
#         return (int(cat_vert_pos), int(cat_horz_pos)), (int(mouse_vert_pos), int(mouse_horz_pos))
#
#
#     def _positions_to_state_index(self, cat_pos, mouse_pos):
#         if not np.prod([ \
#         cat_pos[0] in range(self.board_height), \
#         cat_pos[1] in range(self.board_width), \
#         ]):
#             raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))
#
#         if not np.prod([ \
#         mouse_pos[0] in range(self.board_height), \
#         mouse_pos[1] in range(self.board_width) \
#         ]):
#             raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))
#
#         return int( \
#         cat_pos[0] + \
#         cat_pos[1] * self.board_height  + \
#         mouse_pos[0] * self.board_height * self.board_width + \
#         mouse_pos[1] * self.board_height * self.board_width * self.board_height \
#         )
#
#
#     def _action_to_action_index(self, action):
#         # note that action X means the agent remains where it is
#         valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#         if not action in valid_actions:
#             raise Exception('Invalid action')
#         return int(valid_actions.index(action))
#
#
#     def _action_index_to_action(self, action_index):
#         if not action_index in range(9):
#             raise Exception('action_index must be between 0 and 9')
#         valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#         return valid_actions[action_index]
#
#
#     def _action_to_moves(self, action):
#         # takes action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
#         # returns vert_move, horz_move
#         valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#         if not action in valid_actions:
#             raise Exception('Invalid action')
#         action_index = valid_actions.index(action)
#         horz_move = float(np.mod(action_index, 3) - 1)
#         vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
#         return int(vert_move), int(horz_move)
#
#
#     def _moves_to_action(self, vert_move, horz_move):
#         # takes vert_move, horz_move
#         # returns action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
#         if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
#             raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
#         valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#         return valid_actions[(vert_move + 1 ) * 3 + (horz_move + 1)]
#
#
#     def _action_index_to_moves(self, action_index):
#         if not action_index in range(9):
#             raise Exception('action_index must be between 0 and 9')
#         horz_move = float(np.mod(action_index, 3) - 1)
#         vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
#         return int(vert_move), int(horz_move)
#
#
#     def _moves_to_action_index(vert_move, horz_move):
#         if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
#             raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
#         return int((vert_move + 1 ) * 3 + (horz_move + 1))
