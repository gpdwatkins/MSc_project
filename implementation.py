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

# board_height = 9
# board_width = 9
# walls = [(0,2.5), (3,2.5), (1.5,0), (1.5,2), (1.5, 3), (1.5,5)]

board_height = 9
board_width = 9
walls = [(1,0.5), (1,1.5), (1,2.5), (1,3.5), (1,4.5), (1,5.5), (1,6.5), (1,7.5), \
(3,0.5), (3,1.5), (3,2.5), (3,3.5), (3,4.5), (3,5.5), (3,6.5), (3,7.5), \
(5,0.5), (5,1.5), (5,2.5), (5,3.5), (5,4.5), (5,5.5), (5,6.5), (5,7.5), \
(7,0.5), (7,1.5), (7,2.5), (7,3.5), (7,4.5), (7,5.5), (7,6.5), (7,7.5), \
]

# env = CatMouseEnv_binary_reward(board_height, board_width)
env = CatMouseEnv_proximity_reward(board_height, board_width, walls = walls)

# vvvvvvvvv q_learning training vvvvvvvvv

qvalues_filename, stats_directory_q = train_q_learning(env, no_episodes = 10000000, discount_factor=0.99, alpha=5e-4, eps_start=1, eps_end=0.001, sight = 2, save_stats = True)
policy_q = qvalues_filename
# policy_q = 'trained_parameters/qlearning_qvalues/20190819_1713_4_6_prox_100000000_0.99_0.0005_0.2_inf.txt'

# vvvvvvvvv q_learning training vvvvvvvvv
weights_filename, stats_directory_dqn = train_dqn(env, no_episodes = 1000000, eps_start=1, eps_end=0.001, sight = None, use_belief_state = False, save_stats = True)
policy_dqn = weights_filename
# policy_dqn = 'trained_parameters/dqn_weights/20190821_0057_4_6_prox_1000000_1.0_0.001_0.9999930922685795_inf_False.pth'

weights_filename, stats_directory_drqn = train_drqn(env, no_episodes = 1000000, eps_start=1, eps_end=0.001, sight = float('inf'), save_stats = True)
policy_drqn = weights_filename


policy_q_sight_2 = 'trained_parameters/qlearning_qvalues/20190906_0735_4_6_prox_10000000_0.99_0.0005_1_0.001_None_2.txt'
policy_q_sight_inf = 'trained_parameters/qlearning_qvalues/20190904_0828_4_6_prox_10000000_0.99_0.0005_1_0.001_None_inf.txt'
# policy_q_perf_inf =
policy_dqn_sight_2_no_bs = 'trained_parameters/dqn_weights/20190902_2001_4_6_prox_1000000_1_0.001_None_2_False_dqn.pth'
policy_dqn_sight_2_bs = 'trained_parameters/dqn_weights/20190904_1023_4_6_prox_1000000_1_0.001_None_2_True_dqn.pth'
policy_dqn_sight_inf_no_bs = 'trained_parameters/dqn_weights/20190904_2244_4_6_prox_1000000_1_0.001_None_inf_False_dqn.pth'
policy_dqn_sight_inf_bs = 'trained_parameters/dqn_weights/20190906_0037_4_6_prox_1000000_1_0.001_None_inf_True_dqn.pth'
policy_dqn_perf_inf = 'trained_parameters/dqn_weights/20190906_1214_4_6_prox_1000000_1_0.001_None_None_False_dqn.pth'
policy_drqn_sight_2 = 'trained_parameters/drqn_weights/20190902_2257_4_6_prox_1000000_1_0.001_None_2_False_drqn.pth'
policy_drqn_sight_inf = 'trained_parameters/drqn_weights/20190905_0637_4_6_prox_1000000_1_0.001_None_inf_False_drqn.pth'
# policy_drqn_perf_inf =

stats_q_sight_2 = 'training_analysis/qlearning/20190906_0735_4_6_prox_10000000_0.99_0.0005_1_0.001_None_2'
stats_q_sight_inf = 'training_analysis/qlearning/20190904_0828_4_6_prox_10000000_0.99_0.0005_1_0.001_None_inf'
# stats_q_perf_inf =
stats_dqn_sight_2_no_bs = 'training_analysis/dqn/20190902_2001_4_6_prox_1000000_1_0.001_None_2_False'
stats_dqn_sight_2_bs = 'training_analysis/dqn/20190904_1023_4_6_prox_1000000_1_0.001_None_2_True'
stats_dqn_sight_inf_no_bs = 'training_analysis/dqn/20190904_2244_4_6_prox_1000000_1_0.001_None_inf_False'
stats_dqn_sight_inf_bs = 'training_analysis/dqn/20190906_0037_4_6_prox_1000000_1_0.001_None_inf_True'
stats_dqn_perf_inf = 'training_analysis/dqn/20190906_1214_4_6_prox_1000000_1_0.001_None_None_False'
stats_drqn_sight_2 = 'training_analysis/drqn/20190902_2257_4_6_prox_1000000_1_0.001_None_2_False'
stats_drqn_sight_inf = 'training_analysis/drqn/20190905_0637_4_6_prox_1000000_1_0.001_None_inf_False'
# stats_drqn_perf_inf =

stats_1 = load_training_analysis_from_file(stats_q_sight_2)
stats_2 = load_training_analysis_from_file(stats_dqn_sight_2_no_bs)
stats_3 = load_training_analysis_from_file(stats_dqn_sight_2_bs)
stats_4 = load_training_analysis_from_file(stats_drqn_sight_2)

np.mean(stats_1.episode_lengths[9900:9999])
stats_1.saved_episodes[9999]
np.mean(stats_2.episode_lengths[99000:99999])
stats_2.saved_episodes[99999]
np.mean(stats_3.episode_lengths[99000:99999])
stats_3.saved_episodes[99999]
np.mean(stats_4.episode_lengths[99000:99999])
stats_4.saved_episodes[99999]

# vvvvvvvvv implementation vvvvvvvvv
show_policy((3,5), board_height, board_width, parameter_filename = policy_q, sight = float('inf'), walls = walls)
show_policy((3,5), board_height, board_width, parameter_filename = policy_dqn, walls = walls)
show_policy((3,5), board_height, board_width, parameter_filename = policy_drqn, walls = walls)




# q_test = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_q, seed = 0, sight = float('inf'))
# dqn_test = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn, seed = 0, sight = float('inf'), use_belief_state = False)
# drqn_test = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_drqn, seed = 0, sight = float('inf'))
#
# q_test
# dqn_test
# drqn_test

play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = None, sight = None, use_belief_state = False, seed=1, walls = walls)

play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = policy_q, sight = float('inf'), walls = walls)
play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = policy_dqn, sight = float('inf'), use_belief_state = False, walls = walls)
play_cat_and_mouse(board_height, board_width, show_figs = True, parameter_filename = policy_drqn, sight = float('inf'), walls = None)


# vvvvvvvvv analysis vvvvvvvvv
training_graphical_analysis(stats_directory_q, smoothing_window=1, show_figs = True, save_figs = True)
training_graphical_analysis(stats_directory_dqn, smoothing_window=1, show_figs = True, save_figs = True)
training_graphical_analysis(stats_directory_drqn, smoothing_window=1, show_figs = True, save_figs = True)


compare_training_graphics([stats_dqn_sight_2_no_bs, stats_dqn_sight_2_bs, stats_drqn_sight_2], ['DQN without Belief State', 'DQN with Belief State', 'DRQN'], smoothing_window = 1000)
compare_test_performance([test_dqn_sight_2_no_bs, test_dqn_sight_2_bs, test_drqn_sight_2, test_q_sight_2], ['DQN without Belief State', 'DQN with Belief State', 'DRQN', 'Q-learning'])
# compare_test_performance([test_dqn_sight_2_no_bs, test_dqn_sight_2_bs, test_dqn_perf_inf], ['Sight 2 without Belief State', 'Sight 2 with Belief State', 'Perfect Information'])


# q_test_sight_2 = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_q_sight_2, seed = 0, sight = 2)
# dqn_test_sight_2 = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn_sight_2, seed = 0, sight = 2, use_belief_state = False)
# drqn_test_sight_2 = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_drqn_sight_2, seed = 0, sight = 2)
# q_test_sight_2 =



test_q_sight_2 = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_q_sight_2, seed = 0, sight = 2)
test_q_sight_inf = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_q_sight_inf, seed = 0, sight = float('inf'))
test_q_perf_inf = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_q_perf_inf, seed = 0, sight = None)
test_dqn_sight_2_no_bs = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn_sight_2_no_bs, seed = 0, sight = 2, use_belief_state = False)
test_dqn_sight_2_bs = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn_sight_2_bs, seed = 0, sight = 2, use_belief_state = True)
test_dqn_sight_inf_no_bs = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn_sight_inf_no_bs, seed = 0, sight = float('inf'), use_belief_state = False)
test_dqn_sight_inf_bs = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn_sight_inf_bs, seed = 0, sight = float('inf'), use_belief_state = True)
test_dqn_perf_inf = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_dqn_perf_inf, seed = 0, sight = None)
test_drqn_sight_2 = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_drqn_sight_2, seed = 0, sight = 2)
test_drqn_sight_inf = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_drqn_sight_inf, seed = 0, sight = float('inf'))
test_drqn_perf_inf = test_policy(env, board_height, board_width, 100000, parameter_filename = policy_drqn_perf_inf, seed = 0, sight = None)

test_q_sight_2
test_dqn_sight_2_no_bs
test_dqn_sight_2_bs
test_drqn_sight_2

test_q_sight_inf
test_dqn_sight_inf_no_bs
test_dqn_sight_inf_bs
test_drqn_sight_inf

test_dqn_perf_inf

episodes_to_stabilise_q_sight_2, average_timesteps_q_sight_2 = stabilisation_analysis(stats_q_sight_2, averaging_window = 10, mean_tolerance = 1, var_tolerance = float('inf'))
episodes_to_stabilise_q_sight_2
average_timesteps_q_sight_2

episodes_to_stabilise_dqn_sight_2_no_bs, average_timesteps_dqn_sight_2_no_bs = stabilisation_analysis(stats_dqn_sight_2_no_bs, averaging_window = 1, mean_tolerance = .1, var_tolerance = 1)
episodes_to_stabilise_dqn_sight_2_no_bs
average_timesteps_dqn_sight_2_no_bs

episodes_to_stabilise_dqn_sight_2_bs, average_timesteps_dqn_sight_2_bs = stabilisation_analysis(stats_dqn_sight_2_bs, averaging_window = 1, mean_tolerance = .1, var_tolerance = 1)
episodes_to_stabilise_dqn_sight_2_bs
average_timesteps_dqn_sight_2_bs

episodes_to_stabilise_drqn_sight_2, average_timesteps_drqn_sight_2 = stabilisation_analysis(stats_drqn_sight_2, averaging_window = 1, mean_tolerance = .1, var_tolerance = 1)
episodes_to_stabilise_drqn_sight_2
average_timesteps_drqn_sight_2



stabilisation_q_sight_2 = stabilisation_analysis(stats_q_sight_2, averaging_window = 10, mean_tolerances = [3,2,1,.5,.1,.01])
stabilisation_dqn_sight_2_no_bs = stabilisation_analysis(stats_dqn_sight_2_no_bs, averaging_window = 10, mean_tolerances = [3,2,1,.5,.1,.01])
stabilisation_dqn_sight_2_bs = stabilisation_analysis(stats_dqn_sight_2_bs, averaging_window = 10, mean_tolerances = [3,2,1,.5,.1,.01])
stabilisation_drqn_sight_2 = stabilisation_analysis(stats_drqn_sight_2, averaging_window = 10, mean_tolerances = [3,2,1,.5,.1,.01])

stabilisation_q_sight_2
stabilisation_dqn_sight_2_no_bs
stabilisation_dqn_sight_2_bs
stabilisation_drqn_sight_2
