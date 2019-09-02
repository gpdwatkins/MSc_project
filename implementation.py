from deepqlearning.dqn import train_dqn
from drqn.drqn import train_drqn
from qlearning.qlearning import *
from lib.utils import *
from lib.generate_graphics import *
from lib.generate_gif import play_cat_and_mouse
from lib.analysis import *
from environments.cat_mouse_env import CatMouseEnv_binary_reward, CatMouseEnv_proximity_reward
# import sys
# sys.path.append('C:\\Users\\George\\Anaconda3\\envs\\gym\\Lib\\site-packages')

from importlib import reload
#
# import qlearning.qlearning
# reload(qlearning.qlearning)
# from qlearning.qlearning import *
#
import lib.utils
reload(lib.utils)
from lib.utils import *

import qlearning.qlearning
reload (qlearning.qlearning)
from qlearning.qlearning import *

import deepqlearning.dqn
reload (deepqlearning.dqn)
from deepqlearning.dqn import train_dqn

import drqn.drqn
reload (drqn.drqn)
from drqn.drqn import train_drqn

import lib.generate_graphics
reload (lib.generate_graphics)
from lib.generate_graphics import *

import lib.generate_gif
reload (lib.generate_gif)
from lib.generate_gif import *

import lib.analysis
reload (lib.analysis)
from lib.analysis import *

# vvvvvvvvv parameters vvvvvvvvv
board_height = 4
board_width = 6
walls = [(0,2.5), (3,2.5), (1.5,0), (1.5,2), (1.5, 3), (1.5,5)]
# walls = [(0,2.5), (2,2.5), (3,2.5)]
# env = CatMouseEnv_binary_reward(board_height, board_width)
env = CatMouseEnv_proximity_reward(board_height, board_width, walls = None)

# vvvvvvvvv q_learning training vvvvvvvvv
# policy_filename, stats_directory_q, q_values = train_q_learning(env, no_episodes = 50000, discount_factor=0.99, alpha=5e-4 , epsilon=0.2, sight = float('inf'))

# policy_filename, stats_directory_q, q_values = train_q_learning(env, no_episodes = 50000, discount_factor=0.99, alpha=5e-4 , epsilon=0.2, sight = float('inf'))

# policy_filename, stats_directory_q, q_values = train_q_learning(env, no_episodes = 50000, discount_factor=0.99, alpha=5e-4 , epsilon=0.2, sight = float('inf'))


qvalues_filename, stats_directory_q = train_q_learning(env, no_episodes = 1000, discount_factor=0.99, alpha=5e-4, eps_start=1, eps_end=0.001, sight = None, save_stats = True)
policy_q = qvalues_filename
# policy_q = 'trained_parameters/qlearning_qvalues/20190819_1713_4_6_prox_100000000_0.99_0.0005_0.2_inf.txt'

# vvvvvvvvv q_learning training vvvvvvvvv
weights_filename, stats_directory_dqn = train_dqn(env, no_episodes = 1000, eps_start=1, eps_end=0.001, sight = None, use_belief_state = True, save_stats = True)
policy_dqn = weights_filename
# policy_dqn = 'trained_parameters/dqn_weights/20190821_0057_4_6_prox_1000000_1.0_0.001_0.9999930922685795_inf_False.pth'

weights_filename, stats_directory_drqn = train_drqn(env, no_episodes = 50000, eps_start=1, eps_end=0.001, sight = None, use_belief_state = True, save_stats = True)
policy_drqn = weights_filename

# vvvvvvvvv implementation vvvvvvvvv
show_policy((3,5), board_height, board_width, parameter_filename = policy_q, walls = None)
show_policy((3,5), board_height, board_width, parameter_filename = policy_dqn, walls = None)
show_policy((3,5), board_height, board_width, parameter_filename = policy_drqn, walls = None)


# positions_to_state_index((1,2), (3,5), 4, 6)
#
# print_q(policy_q, 561, board_height, board_width)
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print_q(policy_dqn, 575, board_height, board_width, device = device)
#
#
# print_q(policy_dqn_no_belief, 561, board_height, board_width, device = device)
# print_q(policy_dqn_with_belief, 561, board_height, board_width, device = device)


test_policy(env, board_height, board_width, 1000, parameter_filename = policy_q, seed = 0)
test_policy(env, board_height, board_width, 1000, parameter_filename = policy_dqn, seed = 0)
test_policy(env, board_height, board_width, 1000, parameter_filename = policy_drqn, seed = 0)

play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = None, sight = None, use_belief_state = False, seed=1, walls = walls)

play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = policy_q, sight = None, use_belief_state = False, walls = None)
play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = policy_dqn, sight = None, use_belief_state = True, walls = None)
play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = policy_drqn, sight = None, use_belief_state = True, walls = None)


# vvvvvvvvv analysis vvvvvvvvv
training_graphical_analysis(stats_directory_q, smoothing_window=1, show_figs = True, save_figs = True)
training_graphical_analysis(stats_directory_dqn, smoothing_window=1, show_figs = True, save_figs = True)
training_graphical_analysis(stats_directory_drqn, smoothing_window=1, show_figs = True, save_figs = True)


episodes_to_stabilise_q, average_timesteps_q = stabilisation_analysis(stats_directory_q, averaging_window = 1, mean_tolerance = 5, var_tolerance = 100)
episodes_to_stabilise_q
average_timesteps_q

episodes_to_stabilise_dqn, average_timesteps_dqn = stabilisation_analysis(stats_directory_dqn, averaging_window = 100, mean_tolerance = 5, var_tolerance = 100)
episodes_to_stabilise_dqn
average_timesteps_dqn

episodes_to_stabilise_drqn, average_timesteps_drqn = stabilisation_analysis(stats_directory_drqn, averaging_window = 1, mean_tolerance = 5, var_tolerance = 100)
episodes_to_stabilise_drqn
average_timesteps_drqn




compare_training_graphics([stats_directory_q, stats_directory_dqn])




# stats = load_training_analysis_from_file(stats_directory_q)
# stats = load_training_analysis_from_file(stats_directory_dqn + '/stats.txt')
#
# plot_episode_stats(stats, smoothing_window = 100, show_fig = True)
