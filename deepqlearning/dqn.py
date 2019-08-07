import torch
import numpy as np
import itertools
from collections import deque
from datetime import datetime
from lib.utils import *
# %matplotlib inline
from deepqlearning.dqn_agent import Agent
from collections import namedtuple

# from importlib import reload
# import torch
# reload(torch)
# import torch

# def dqn(env, no_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
#     """Deep Q-Learning.
#
#     Params
#     ======
#         no_episodes (int): maximum number of training episodes
#         max_t (int): maximum number of timesteps per episode
#         eps_start (float): starting value of epsilon, for epsilon-greedy action selection
#         eps_end (float): minimum value of epsilon
#         eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
#     """
#     env.seed(0)
#     board_height = env.board_height
#     board_width = env.board_width
#
#     stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
#         episode_lengths=np.zeros(no_episodes),
#         episode_rewards=np.zeros(no_episodes))
#
#     start_time = datetime.now().strftime('%Y%m%d_%H%M')
#
#     # filename = 'trained_weights/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(max_t), str(eps_start), str(eps_end), str(eps_decay)]) + '.pth'
#     filename = 'trained_weights/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(max_t), str(eps_start), str(eps_end), str(eps_decay)]) + '.pth'
#
#     scores_window = deque(maxlen=100)  # last 100 scores
#     eps = eps_start                    # initialize epsilon
#     agent = Agent(state_size = 2 * env.board_height * env.board_width, action_size = 9, seed=0)
#     for episode in range(1, no_episodes+1):
#         state_index = env.reset()
#         cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)
#         state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
#         score = 0
#
#         for timestep in itertools.count():
#         # for t in range(max_t):
#             action = agent.act(state, eps)
#             next_state_index, reward, done, _ = env.step(action)
#             next_cat_pos, next_mouse_pos = state_index_to_positions(next_state_index, board_height, board_width)
#             next_state = positions_to_nn_input(next_cat_pos, next_mouse_pos, board_height, board_width)
#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             score += reward
#             if done:
#                 stats.episode_lengths[episode-1] = timestep
#                 break
#         stats.episode_rewards[episode-1] = score
#
#         scores_window.append(score)       # save most recent score
#         # scores.append(score)              # save most recent score
#         eps = max(eps_end, eps_decay*eps) # decrease epsilon
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
#         if episode % 100 == 0:
#             print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
#         # if len(scores_window) == 100 and np.mean(scores_window) >= board_height * board_width - min(board_height, board_width):
#         #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
#         #     torch.save(agent.qnetwork_behaviour.state_dict(), filename)
#         #     stats = stats._replace(episode_rewards = stats.episode_rewards[:episode], episode_lengths = stats.episode_lengths[:episode])
#         #     break
#         torch.save(agent.qnetwork_behaviour.state_dict(), filename)
#     return filename, stats


def train_dqn(env, no_episodes, max_t=1000, eps_start=1.0, eps_end=0.001, eps_decay=None, sight = 2, use_belief_state = True):
    """Deep Q-Learning.

    Params
    ======
        no_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    env.seed(0)
    board_height = env.board_height
    board_width = env.board_width

    if eps_decay == None:
        eps_decay = (eps_end/eps_start)**(1/no_episodes)

    stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
        episode_lengths=np.zeros(no_episodes),
        episode_rewards=np.zeros(no_episodes))

    start_time = datetime.now().strftime('%Y%m%d_%H%M')

    # filename = 'trained_weights/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(max_t), str(eps_start), str(eps_end), str(eps_decay)]) + '.pth'
    filename = 'trained_parameters/dqn_weights/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(eps_start), str(eps_end), str(eps_decay), str(sight), str(use_belief_state)]) + '.pth'

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    agent = Agent(state_size = 2 * env.board_height * env.board_width, action_size = 9, seed=0)
    for episode in range(1, no_episodes+1):
        state_index = env.reset()
        cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)

        # initial_cat_distribution = np.zeros(board_height * board_width)
        # initial_mouse_distribution = np.ones(board_height * board_width) * (1/(board_height * board_width))
        initial_state_distribution = np.ones(2 * board_height * board_width)
        # np.concatenate((initial_cat_distribution, initial_mouse_distribution))
        # observed_state_nn_input = observed_state(true_state_index, np.ones(board_height * board_width), board_height, board_width)
        observed_state_nn_input = observed_state_as_nn_input(board_height, board_width, state_index, initial_state_distribution, sight = sight, use_belief_state = use_belief_state)

        # state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
        score = 0

        for timestep in itertools.count():
        # for t in range(max_t):
            action = agent.act(observed_state_nn_input, eps)
            next_state_index, reward, done, _ = env.step(action)
            next_observed_state_nn_input = observed_state_as_nn_input(board_height, board_width, next_state_index, observed_state_nn_input, sight = sight, use_belief_state = use_belief_state)


            # next_cat_pos, next_mouse_pos = state_index_to_positions(next_state_index, board_height, board_width)
            # next_state = positions_to_nn_input(next_cat_pos, next_mouse_pos, board_height, board_width)
            agent.step(observed_state_nn_input, action, reward, next_observed_state_nn_input, done)
            observed_state_nn_input = next_observed_state_nn_input
            score += reward
            if done:
                stats.episode_lengths[episode-1] = timestep
                break
        stats.episode_rewards[episode-1] = score

        scores_window.append(score)       # save most recent score
        # scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        # if len(scores_window) == 100 and np.mean(scores_window) >= board_height * board_width - min(board_height, board_width):
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        #     torch.save(agent.qnetwork_behaviour.state_dict(), filename)
        #     stats = stats._replace(episode_rewards = stats.episode_rewards[:episode], episode_lengths = stats.episode_lengths[:episode])
        #     break
        torch.save(agent.qnetwork_behaviour.state_dict(), filename)
    return filename, stats
