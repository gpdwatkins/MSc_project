import numpy as np
import gym

# from gym import Env, spaces
# from gym.utils import seeding

def categorical_sample(prob_n, np_random = None):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    # print(prob_n)
    csprob_n = np.cumsum(prob_n)
    # print(csprob_n)
    return (csprob_n > np.random.rand()).argmax()



# def initialise_cat_mouse_positions(board_height, board_width):
#     # board_height, board_width = np.shape(board)
#     cat_pos = (np.random.randint(0,board_height), np.random.randint(0,board_width))
#     mouse_pos = (np.random.randint(0,board_height), np.random.randint(0,board_width))
#     while mouse_pos == cat_pos:
#         mouse_pos = (np.random.randint(0,board_height), np.random.randint(0,board_width))
#     return cat_pos, mouse_pos
#
# def state_index_to_positions(state_index, board_height, board_width):
#     if not state_index in range(board_height**2 * board_width**2):
#         raise ValueError('state_index must be between 0 and %d' % (board_height**2 * board_width**2 - 1))
#
#     cat_vert_pos = float(np.mod(state_index,board_height))
#     cat_horz_pos = np.mod(state_index - cat_vert_pos, board_height * board_width) / board_height
#     mouse_vert_pos = np.mod(state_index - cat_horz_pos * board_height - cat_vert_pos, board_height * board_width * board_height) / (board_height * board_width)
#     mouse_horz_pos = (state_index - mouse_vert_pos * board_height * board_width - cat_horz_pos * board_height - cat_vert_pos) / (board_height * board_width * board_height)
#     return (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos)
#
#
# def positions_to_state_index(cat_pos, mouse_pos, board_height, board_width):
#     if not np.prod([ \
#     cat_pos[0] in range(board_height), \
#     cat_pos[1] in range(board_width), \
#     ]):
#         raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))
#
#     if not np.prod([ \
#     mouse_pos[0] in range(board_height), \
#     mouse_pos[1] in range(board_width) \
#     ]):
#         raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))
#
#     return cat_pos[0] + \
#     cat_pos[1] * board_height  + \
#     mouse_pos[0] * board_height * board_width + \
#     mouse_pos[1] * board_height * board_width * board_height
#
#
# def action_to_action_index(action):
#     # note that action X means the agent remains where it is
#     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#     if not action in valid_actions:
#         raise Exception('Invalid action')
#     return valid_actions.index(action)
#
#
# def action_index_to_action(action_index):
#     if not action_index in range(9):
#         raise Exception('action_index must be between 0 and 9')
#     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#     return valid_actions[action_index]
#
#
# def action_to_moves(action):
#     # takes action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
#     # returns vert_move, horz_move
#     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#     if not action in valid_actions:
#         raise Exception('Invalid action')
#     action_index = valid_actions.index(action)
#     horz_move = float(np.mod(action_index, 3) - 1)
#     vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
#     return vert_move, horz_move
#
#
# def moves_to_action(vert_move, horz_move):
#     # takes vert_move, horz_move
#     # returns action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
#     if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
#         raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
#     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#     return valid_actions[(vert_move + 1 ) * 3 + (horz_move + 1)]
#
# def action_index_to_moves(action_index):
#     if not action_index in range(9):
#         raise Exception('action_index must be between 0 and 9')
#     horz_move = float(np.mod(action_index, 3) - 1)
#     vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
#     return vert_move, horz_move
#
# def moves_to_action_index(vert_move, horz_move):
#     valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
#     if not action in valid_actions:
#         raise Exception('Invalid action')
#     return (vert_move + 1 ) * 3 + (horz_move + 1)


import numpy as np
import sys
from gym.envs.toy_text import discrete

# UP = 0
# RIGHT = 1
# DOWN = 2
# LEFT = 3
actions = ['NW','N','NE','W','X','E','SW','S','SE']

# class CliffWalkingEnv(discrete.DiscreteEnv):
class CatMouseEnv(discrete.DiscreteEnv):
    # """
    # This is a simple implementation of the Gridworld Cliff
    # reinforcement learning task.
    #
    # Adapted from Example 6.6 (page 106) from Reinforcement Learning: An Introduction
    # by Sutton and Barto:
    # http://incompleteideas.net/book/bookdraft2018jan1.pdf
    #
    # With inspiration from:
    # https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py
    #
    # The board is a 4x12 matrix, with (using Numpy matrix indexing):
    #     [3, 0] as the start at bottom-left
    #     [3, 11] as the goal at bottom-right
    #     [3, 1..10] as the cliff at bottom-center
    #
    # Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward
    # and a reset to the start. An episode terminates when the agent reaches the goal.
    # """
    metadata = {'render.modes': ['human', 'ansi']}


    def __init__(self):
        # self.shape = (4, 12)
        self.board_height = 4
        self.board_width = 6
        # self.start_state_index = np.ravel_multi_index((3, 0), self.shape)


        # nS = np.prod(self.shape)
        nS = self.board_height * self.board_width * self.board_height * self.board_width
        # nA = 4
        nA = len(actions)

        # # Cliff Location
        # self._cliff = np.zeros(self.shape, dtype=np.bool)
        # self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        # P = {}
        # for s in range(nS):
        #     position = np.unravel_index(s, self.shape)
        #     P[s] = {a: [] for a in range(nA)}
        #     P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
        #     P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
        #     P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
        #     P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        P = {}
        for s in range(nS):
            (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = self._state_index_to_positions(s)
        # for cat_vert_pos in range(board_height):
        #     for cat_horz_pos in range(board_width):
        #         for mouse_vert_pos in range(board_height):
        #             for mouse_horz_pos in range(board_width):
            state_index = self._positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos))
            P[state_index] = {}
            for action in actions:
                cat_vert_move, cat_horz_move = self._action_to_moves(action)
                cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(self.board_height)) and ((cat_horz_pos + cat_horz_move) in range(self.board_width))
                new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
                new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
                new_state_instances = {}
                mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
                for mouse_vert_move in mouse_vert_moves:
                    for mouse_horz_move in mouse_horz_moves:
                        mouse_action = self._moves_to_action(mouse_vert_move, mouse_horz_move)
                        mouse_move_stays_on_board = ((mouse_vert_pos + mouse_vert_move) in range(self.board_height)) and ((mouse_horz_pos + mouse_horz_move) in range(self.board_width))
                        new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
                        new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
                        new_state = self._positions_to_state_index((new_cat_vert_pos, new_cat_horz_pos), (new_mouse_vert_pos, new_mouse_horz_pos))
                        if not new_state in new_state_instances.keys():
                            game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
                            new_state_instances[new_state] = [1, game_over]
                        else:
                            new_state_instances[new_state][0] += 1
                        # if (action == 'NW') and (cat_vert_pos == 5) and (cat_horz_pos == 0) and (mouse_vert_pos == 0) and (mouse_horz_pos == 3):
                        #     print('mouse_action: (%d, %d)' % (mouse_vert_move, mouse_horz_move))
                        #     print('new_mouse_location: (%d, %d)' % (new_mouse_vert_pos, new_mouse_horz_pos))
                possible_next_states_list = []
                for new_state, (no_instances, game_over) in new_state_instances.items():
                    possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, 1 * game_over, game_over)
                    possible_next_states_list.append(possible_next_state_tuple)
                P[state_index][self._action_to_action_index(action)] = possible_next_states_list






        # Calculate initial state distribution
        # We always start in state (3, 0)
        # isd = np.zeros(nS)
        # isd[self.start_state_index] = 1.0

        isd = np.ones(nS)
        for state_index in range(nS):
            (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = self._state_index_to_positions(state_index)
            if ((cat_vert_pos == mouse_vert_pos) and (cat_horz_pos == mouse_horz_pos)):
                isd[state_index] = 0
        isd = isd/sum(isd)



        super(CatMouseEnv, self).__init__(nS, nA, P, isd)

    # def _limit_coordinates(self, coord):
    #     """
    #     Prevent the agent from falling out of the grid world
    #     :param coord:
    #     :return:
    #     """
    #     coord[0] = min(coord[0], self.shape[0] - 1)
    #     coord[0] = max(coord[0], 0)
    #     coord[1] = min(coord[1], self.shape[1] - 1)
    #     coord[1] = max(coord[1], 0)
    #     return coord
    #
    # def _calculate_transition_prob(self, current, delta):
    #     """
    #     Determine the outcome for an action. Transition Prob is always 1.0.
    #     :param current: Current position on the grid as (row, col)
    #     :param delta: Change in position for transition
    #     :return: (1.0, new_state, reward, done)
    #     """
    #     new_position = np.array(current) + np.array(delta)
    #     new_position = self._limit_coordinates(new_position).astype(int)
    #     new_state = np.ravel_multi_index(tuple(new_position), self.shape)
    #     if self._cliff[tuple(new_position)]:
    #         return [(1.0, self.start_state_index, -100, False)]
    #
    #     terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
    #     is_done = tuple(new_position) == terminal_state
    #     return [(1.0, new_state, -1, is_done)]

    def render(self, mode='human'):
        outfile = sys.stdout

        for row in range(self.board_height):
            for col in range(self.board_width):
                (cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos) = self._state_index_to_positions(self.s)

        # for s in range(self.nS):
            # position = np.unravel_index(s, self.shape)

                # if self.s == s:
                # output = " x "
                # # Print terminal state
                # elif position == (3, 11):
                #     output = " T "
                # elif self._cliff[position]:
                #     output = " C "
                # else:
                #     output = " o "
                if (cat_vert_pos == row and cat_horz_pos == col and mouse_vert_pos == row and mouse_horz_pos == col):
                    output = " @ "
                elif (cat_vert_pos == row and cat_horz_pos == col):
                    output = " C "
                elif (mouse_vert_pos == row and mouse_horz_pos == col):
                    output = " M "
                else:
                    output = " o "

            # if position[1] == 0:
            #     output = output.lstrip()
            # if position[1] == self.shape[1] - 1:
            #     output = output.rstrip()
            #     output += '\n'
                if col == 0:
                    output = output.lstrip()
                if col == (self.board_width - 1):
                    output = output.rstrip()
                    output += '\n'
                outfile.write(output)
            outfile.write('\n')


    def _state_index_to_positions(self, state_index):
        if not state_index in range(self.board_height**2 * self.board_width**2):
            raise ValueError('state_index must be between 0 and %d' % (self.board_height**2 * self.board_width**2 - 1))

        cat_vert_pos = float(np.mod(state_index,self.board_height))
        cat_horz_pos = np.mod(state_index - cat_vert_pos, self.board_height * self.board_width) / self.board_height
        mouse_vert_pos = np.mod(state_index - cat_horz_pos * self.board_height - cat_vert_pos, self.board_height * self.board_width * self.board_height) / (self.board_height * self.board_width)
        mouse_horz_pos = (state_index - mouse_vert_pos * self.board_height * self.board_width - cat_horz_pos * self.board_height - cat_vert_pos) / (self.board_height * self.board_width * self.board_height)
        return (int(cat_vert_pos), int(cat_horz_pos)), (int(mouse_vert_pos), int(mouse_horz_pos))


    def _positions_to_state_index(self, cat_pos, mouse_pos):
        if not np.prod([ \
        cat_pos[0] in range(self.board_height), \
        cat_pos[1] in range(self.board_width), \
        ]):
            raise ValueError('Cat position (%d,%d) is outside the board' % (cat_pos[0], cat_pos[1]))

        if not np.prod([ \
        mouse_pos[0] in range(self.board_height), \
        mouse_pos[1] in range(self.board_width) \
        ]):
            raise ValueError('Mouse position (%d,%d) is outside the board' % (mouse_pos[0], mouse_pos[1]))

        return int( \
        cat_pos[0] + \
        cat_pos[1] * self.board_height  + \
        mouse_pos[0] * self.board_height * self.board_width + \
        mouse_pos[1] * self.board_height * self.board_width * self.board_height \
        )


    def _action_to_action_index(self, action):
        # note that action X means the agent remains where it is
        valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
        if not action in valid_actions:
            raise Exception('Invalid action')
        return int(valid_actions.index(action))


    def _action_index_to_action(self, action_index):
        if not action_index in range(9):
            raise Exception('action_index must be between 0 and 9')
        valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
        return valid_actions[action_index]


    def _action_to_moves(self, action):
        # takes action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
        # returns vert_move, horz_move
        valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
        if not action in valid_actions:
            raise Exception('Invalid action')
        action_index = valid_actions.index(action)
        horz_move = float(np.mod(action_index, 3) - 1)
        vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
        return int(vert_move), int(horz_move)


    def _moves_to_action(self, vert_move, horz_move):
        # takes vert_move, horz_move
        # returns action from ['NW','N','NE','W','X','E','SW','S','SE'] as input
        if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
            raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
        valid_actions = ['NW','N','NE','W','X','E','SW','S','SE']
        return valid_actions[(vert_move + 1 ) * 3 + (horz_move + 1)]


    def _action_index_to_moves(self, action_index):
        if not action_index in range(9):
            raise Exception('action_index must be between 0 and 9')
        horz_move = float(np.mod(action_index, 3) - 1)
        vert_move = np.mod((action_index - horz_move - 1)/3, 3) - 1
        return int(vert_move), int(horz_move)


    def moves_to_action_index(vert_move, horz_move):
        if not (vert_move in [-1, 0, 1] and horz_move in [-1, 0, 1]):
            raise Exception('Invalid move. Vertical and horizontal components must be in [-1. 0. 1]')
        return int((vert_move + 1 ) * 3 + (horz_move + 1))






















# class DiscreteEnv(Env):
class Cat_mouse_env():

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
#     def __init__(self, nS, nA, P, isd):
#         self.P = P
#         self.isd = isd
#         self.lastaction = None # for rendering
#         self.nS = nS
#         self.nA = nA
#
#         self.action_space = spaces.Discrete(self.nA)
#         self.observation_space = spaces.Discrete(self.nS)
#
#         self.seed()
#         self.s = categorical_sample(self.isd, self.np_random)
#         self.lastaction=None

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None # for rendering
        self.nS = nS
        self.nA = nA

#         self.action_space = spaces.Discrete(self.nA)
#         self.observation_space = spaces.Discrete(self.nS)

        self.action_space = range(self.nA)
        self.observation_space = range(self.nS)


#         self.seed()
#         self.s = categorical_sample(self.isd, self.np_random)
        self.s = categorical_sample(self.isd)

#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

#     def reset(self):
#         self.s = categorical_sample(self.isd, self.np_random)
#         self.lastaction = None
#         return self.s

    def reset(self):
        elf.s = categorical_sample(self.isd)
        self.lastaction = None
        return self.s

#     def step(self, a):
#         transitions = self.P[self.s][a]
#         i = categorical_sample([t[0] for t in transitions], self.np_random)
#         p, s, r, d= transitions[i]
#         self.s = s
#         self.lastaction = a
#         return (s, r, d, {"prob" : p})

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions])
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})





# ### vvv GENERATE TRANSITION MATRIX vvv ###
#
# # dict of dicts of lists
# # first dict is keyed on states
# # second dict is keyed on actions
# # lists are of the form  [(probability, nextstate, reward, done), ...]
# # lists contain tuples
# # each tuple is a possible consequence of taking that state-action
# # if the system is deterministic, each list has only one element (a tuple with first entry 1)
#
# # iterate through states (i.e. cat_pos and mouse_pos)
# # iterate through all 9 actions
#
# board_height = 6
# board_width = 4
# actions = ['NW','N','NE','W','X','E','SW','S','SE']
# scratch_transition_matrix = {}
# for cat_vert_pos in range(board_height):
#     for cat_horz_pos in range(board_width):
#         for mouse_vert_pos in range(board_height):
#             for mouse_horz_pos in range(board_width):
#                 state_index = positions_to_state_index((cat_vert_pos, cat_horz_pos), (mouse_vert_pos, mouse_horz_pos), board_height, board_width)
#                 scratch_transition_matrix[state_index] = {}
#                 for action in actions:
#                     cat_vert_move, cat_horz_move = action_to_moves(action)
#                     cat_move_stays_on_board = ((cat_vert_pos + cat_vert_move) in range(board_height)) and ((cat_horz_pos + cat_horz_move) in range(board_width))
#                     new_cat_vert_pos = cat_vert_pos + cat_vert_move * cat_move_stays_on_board
#                     new_cat_horz_pos = cat_horz_pos + cat_horz_move * cat_move_stays_on_board
#                     new_state_instances = {}
#                     mouse_vert_moves, mouse_horz_moves = [-1,0,1], [-1,0,1]
#                     for mouse_vert_move in mouse_vert_moves:
#                         for mouse_horz_move in mouse_horz_moves:
#                             mouse_action = moves_to_action(mouse_vert_move, mouse_horz_move)
#                             mouse_move_stays_on_board = ((mouse_vert_pos + mouse_vert_move) in range(board_height)) and ((mouse_horz_pos + mouse_horz_move) in range(board_width))
#                             new_mouse_vert_pos = mouse_vert_pos + mouse_vert_move * mouse_move_stays_on_board
#                             new_mouse_horz_pos = mouse_horz_pos + mouse_horz_move * mouse_move_stays_on_board
#                             new_state = positions_to_state_index((new_cat_vert_pos, new_cat_horz_pos), (new_mouse_vert_pos, new_mouse_horz_pos), board_height, board_width)
#                             if not new_state in new_state_instances.keys():
#                                 game_over = ((new_cat_vert_pos == new_mouse_vert_pos) and (new_cat_horz_pos == new_mouse_horz_pos))
#                                 new_state_instances[new_state] = [1, game_over]
#                             else:
#                                 new_state_instances[new_state][0] += 1
#                             # if (action == 'NW') and (cat_vert_pos == 5) and (cat_horz_pos == 0) and (mouse_vert_pos == 0) and (mouse_horz_pos == 3):
#                             #     print('mouse_action: (%d, %d)' % (mouse_vert_move, mouse_horz_move))
#                             #     print('new_mouse_location: (%d, %d)' % (new_mouse_vert_pos, new_mouse_horz_pos))
#                     possible_next_states_list = []
#                     for new_state, (no_instances, game_over) in new_state_instances.items():
#                         possible_next_state_tuple = (no_instances/(len(mouse_vert_moves) * len(mouse_horz_moves)), new_state, 1 * game_over, game_over)
#                         possible_next_states_list.append(possible_next_state_tuple)
#                     scratch_transition_matrix[state_index][action_to_action_index(action)] = possible_next_states_list



# env = gym.make('Cat_mouse_env')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()
