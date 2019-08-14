from deepqlearning.dqn import train_dqn
from qlearning.qlearning import *
from lib.utils import *
from lib.generate_graphics import *
from lib.generate_gif import play_cat_and_mouse
from lib.analysis import *
from environments.cat_mouse_env import CatMouseEnv_binary_reward, CatMouseEnv_proximity_reward
# import sys
# sys.path.append('C:\\Users\\George\\Anaconda3\\envs\\gym\\Lib\\site-packages')

from importlib import reload

import qlearning.qlearning
reload(qlearning.qlearning)
from qlearning.qlearning import *


# vvvvvvvvv parameters vvvvvvvvv
board_height = 4
board_width = 6
# env = CatMouseEnv_binary_reward(board_height, board_width)
env = CatMouseEnv_proximity_reward(board_height, board_width)


# vvvvvvvvv q_learning training vvvvvvvvv
policy_filename, stats_directory_q = train_q_learning(env, no_episodes = 500000, discount_factor=0.99, alpha=5e-4 , epsilon=0.1, sight = float('inf'))
policy_q = policy_filename


# vvvvvvvvv q_learning training vvvvvvvvv
weights_filename, stats_directory_dqn = train_dqn(env, no_episodes = 500000, sight = float('inf'), use_belief_state = False)
policy_dqn = weights_filename
# policy = '20190729_1926_4_6_prox_10000_1000_1.0_0.01_0.9995.pth'


# extract_training_metadata(policy_filename)

# vvvvvvvvv implementation vvvvvvvvv
show_policy((3,5), board_height, board_width, policy = policy_q)
show_policy((3,5), board_height, board_width, policy = policy_dqn)

test_policy(env, board_height, board_width, 10000, policy = policy_q, seed = 0)
test_policy(env, board_height, board_width, 10000, policy = policy_dqn, seed = 0)

play_cat_and_mouse(board_height, board_width, show_figs = True, policy = None, sight = float('inf'), use_belief_state = True, seed=1)
play_cat_and_mouse(board_height, board_width, show_figs = True, policy = policy_q, sight = 2, use_belief_state = True)
play_cat_and_mouse(board_height, board_width, show_figs = True, policy = policy_dqn, sight = 2, use_belief_state = True)

# vvvvvvvvv analysis vvvvvvvvv
training_graphical_analysis(stats_directory_q, smoothing_window=100, show_figs = True, save_figs = True)
training_graphical_analysis(stats_directory_dqn, smoothing_window=100, show_figs = True, save_figs = True)

# stats_q = load_training_analysis_from_file(stats_directory_q + '/stats.txt')
# stats_dqn = load_training_analysis_from_file(stats_directory_dqn + '/stats.txt')

# plot_episode_stats(stats_q)
# plot_episode_stats(stats_dqn)

# stats_directory_q
episodes_to_stabilise_q, average_timesteps_q = stabilisation_analysis(stats_directory_q, averaging_window = 100, mean_tolerance = 1, var_tolerance = 30)
episodes_to_stabilise_q
average_timesteps_q

episodes_to_stabilise_dqn, average_timesteps_dqn = stabilisation_analysis(stats_directory_dqn, averaging_window = 100, mean_tolerance = 1, var_tolerance = 30)
episodes_to_stabilise_dqn
average_timesteps_dqn

compare_training_graphics([stats_directory_q, stats_directory_dqn])
