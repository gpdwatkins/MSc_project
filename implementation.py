from deepqlearning.dqn import train_dqn
from qlearning.q_learning import *
from lib.utils import *
from lib.generate_gif import play_cat_and_mouse
from environments.cat_mouse_env import CatMouseEnv_binary_reward, CatMouseEnv_proximity_reward
from lib.generate_graphics import *
# import sys
# sys.path.append('C:\\Users\\George\\Anaconda3\\envs\\gym\\Lib\\site-packages')

from importlib import reload

# import lib.generate_gif
# reload(lib.generate_gif)
# from lib.generate_gif import play_cat_and_mouse


# vvvvvvvvv parameters vvvvvvvvv
board_height = 4
board_width = 6
# env = CatMouseEnv_binary_reward(board_height, board_width)
env = CatMouseEnv_proximity_reward(board_height, board_width)


# vvvvvvvvv q_learning training vvvvvvvvv
policy_filename, stats_q = train_q_learning(env, no_episodes = 100000, discount_factor=0.5, alpha=0.2, epsilon=0.2, sight = float('inf'))
policy_q = policy_filename


# vvvvvvvvv q_learning training vvvvvvvvv
weights_filename, stats_dqn = train_dqn(env, no_episodes = 10000, sight = float('inf'), use_belief_state = True)
policy_dqn = weights_filename
# policy = '20190729_1926_4_6_prox_10000_1000_1.0_0.01_0.9995.pth'


# vvvvvvvvv implementation vvvvvvvvv
show_policy((3,5), board_height, board_width, policy = policy_q)
show_policy((3,5), board_height, board_width, policy = policy_dqn)

test_policy(env, board_height, board_width, 10000, policy = policy_q, seed = None)
test_policy(env, board_height, board_width, 10000, policy = policy_dqn, seed = None)

play_cat_and_mouse(board_height, board_width, show_figs = True, policy = policy_q, sight = 2, use_belief_state = True)
play_cat_and_mouse(board_height, board_width, show_figs = True, policy = policy_dqn, sight = 2, use_belief_state = True)
play_cat_and_mouse(board_height, board_width, show_figs = True, policy = None, sight = 2, use_belief_state = True, seed=1)

# vvvvvvvvv analysis vvvvvvvvv
plot_episode_stats(stats_q)
plot_episode_stats(stats_dqn)

episodes_to_stabilise_q, average_timesteps_q = stabilisation_analysis(stats_q, averaging_window = 100, mean_tolerance = 1, var_tolerance = 100)
episodes_to_stabilise_q
average_timesteps_q

episodes_to_stabilise_dqn, average_timesteps_dqn = stabilisation_analysis(stats_dqn, averaging_window = 100, mean_tolerance = 1, var_tolerance = 100)
episodes_to_stabilise_dqn
average_timesteps_dqn
