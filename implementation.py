import sys
sys.path.append('C:\\Users\\George\\Anaconda3\\envs\\gym\\Lib\\site-packages')
from deepqlearning.dqn import dqn
from qlearning.q_learning import *
from lib.utils import *
from lib.generate_gif import play_cat_and_mouse
from environments.cat_mouse_env import CatMouseEnv_binary_reward, CatMouseEnv_proximity_reward
from lib.generate_graphics import *

from importlib import reload

# test comment

# vvvvvvvvv parameters vvvvvvvvv
board_height = 4
board_width = 6
# env = CatMouseEnv_binary_reward(board_height, board_width)
env = CatMouseEnv_proximity_reward(board_height, board_width)


# vvvvvvvvv q_learning training vvvvvvvvv
Q_values, stats = q_learning(env, 100000)
policy = get_greedy_policy(Q_values)


# vvvvvvvvv q_learning training vvvvvvvvv
weights_file, stats_dqn = dqn(env)
policy = weights_file
# policy = '20190729_1926_4_6_prox_10000_1000_1.0_0.01_0.9995.pth'


# vvvvvvvvv implementation vvvvvvvvv
show_policy((2,0), board_height, board_width, policy = policy)


play_cat_and_mouse(board_height, board_width, show_figs = True, policy = policy, sight = 2, mouse_pos_dist = True)


# vvvvvvvvv analysis vvvvvvvvv
plot_episode_stats(stats_dqn)

episodes_to_stabilise, average_timesteps = stabilisation_analysis(stats_dqn, averaging_window = 100, mean_tolerance = 1, var_tolerance = 100)
episodes_to_stabilise
average_timesteps
