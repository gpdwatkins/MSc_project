# import gym
# import random
import torch
import numpy as np
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


def dqn(env, no_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.9995):
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

    stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
        episode_lengths=np.zeros(no_episodes),
        episode_rewards=np.zeros(no_episodes))

    start_time = datetime.now().strftime('%Y%m%d_%H%M')

    filename = 'trained_weights/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(max_t), str(eps_start), str(eps_end), str(eps_decay)]) + '.pth'

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    agent = Agent(state_size = 2 + env.board_height * env.board_width, action_size = 9, seed=0)
    for episode in range(1, no_episodes+1):
        state_index = env.reset()
        cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)
        state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state_index, reward, done, _ = env.step(action)
            next_cat_pos, next_mouse_pos = state_index_to_positions(next_state_index, board_height, board_width)
            next_state = positions_to_nn_input(next_cat_pos, next_mouse_pos, board_height, board_width)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
            # if episode % 100 == 0:
        stats.episode_rewards[episode-1] = score
        stats.episode_lengths[episode-1] = t
        scores_window.append(score)       # save most recent score
        # scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if len(scores_window) == 100 and np.mean(scores_window) >= board_height * board_width - min(board_height, board_width):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_behaviour.state_dict(), filename)
            stats = stats._replace(episode_rewards = stats.episode_rewards[:episode], episode_lengths = stats.episode_lengths[:episode])
            break
    return filename, stats
