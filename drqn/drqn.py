import torch
import numpy as np
import itertools
from collections import deque
from datetime import datetime
from os import mkdir
from lib.utils import *
# %matplotlib inline
from drqn.drqn_agent import DRQNAgent
from collections import namedtuple


def drqn(env, weights_filename, no_episodes, eps_start, eps_end, eps_decay, sight):
    """Deep Q-Learning.

    Params
    ======
        no_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    flag = 0


    # code adapted from https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/Deep_Q_Network.ipynb
    # https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/Deep_Q_Network.ipynb
    env.seed(0)
    board_height = env.board_height
    board_width = env.board_width
    walls = env.walls

    if eps_decay == None:
        eps_decay = (eps_end/eps_start)**(1/no_episodes)

    max_stats_datapoints = 100000
    save_stats_every = int(no_episodes / max_stats_datapoints) + 1 * (no_episodes % max_stats_datapoints != 0)
    stats = namedtuple("Stats",["saved_episodes", "episode_lengths", "episode_rewards"])(
        saved_episodes=[],
        episode_lengths=np.zeros(min(max_stats_datapoints,no_episodes)),
        episode_rewards=np.zeros(min(max_stats_datapoints,no_episodes)))

    running_episode_rewards = 0
    running_episode_lengths = 0

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    drqn_agent = DRQNAgent(state_size = 2 * env.board_height * env.board_width, action_size = 9, seed=0)
    for episode in range(1, no_episodes+1):
        state_index = env.reset()
        cat_pos, mouse_pos = state_index_to_positions(state_index, board_height, board_width)

        initial_state_distribution = np.ones(2 * board_height * board_width)
        observed_state_nn_input = observed_state_as_nn_input(board_height, board_width, state_index, initial_state_distribution, sight = sight, walls = walls)
        observed_state_drqn_input = nn_input_as_drqn_input(board_height, board_width, observed_state_nn_input)

        hidden = drqn_agent.init_hidden()

        score = 0

        for timestep in itertools.count():
            action, next_hidden = drqn_agent.act(observed_state_drqn_input, hidden, eps)
            next_state_index, reward, done, _ = env.step(action)
            # if flag < 2:
            #     print(next_state_index)
            #     flag += 1
            next_observed_state_nn_input = observed_state_as_nn_input(board_height, board_width, next_state_index, observed_state_nn_input, sight = sight, walls = walls)
            next_observed_state_drqn_input = nn_input_as_drqn_input(board_height, board_width, next_observed_state_nn_input)

            drqn_agent.step(timestep == 0, observed_state_drqn_input, hidden, action, reward, next_observed_state_drqn_input, next_hidden, done)
            observed_state_nn_input = next_observed_state_nn_input
            observed_state_drqn_input = next_observed_state_drqn_input
            hidden = next_hidden
            score += reward
            if done:
                running_episode_lengths += timestep + 1
                running_episode_rewards += score
                if episode % save_stats_every == 0:
                    stats.saved_episodes.append(episode)
                    stats.episode_lengths[int(episode/save_stats_every) - 1] = running_episode_lengths/save_stats_every
                    stats.episode_rewards[int(episode/save_stats_every) - 1] = running_episode_rewards/save_stats_every
                    running_episode_rewards = 0
                    running_episode_lengths = 0
                break
        # if episode % save_stats_every == 0:
        #     stats.episode_rewards[int(episode/save_stats_every) - 1] = score

        scores_window.append(score)       # save most recent score
        # scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 10000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))

    torch.save(drqn_agent.drqn_behaviour.state_dict(), weights_filename)
    # if flag == 0:
    #     print(drqn_agent.drqn_behaviour.state_dict())
    return stats


def train_drqn(env, no_episodes, eps_start=1.0, eps_end=0.001, eps_decay=None, sight = float('inf'), use_belief_state = False, save_weights = True, save_stats = True):
    start_time = datetime.now().strftime('%Y%m%d_%H%M')
    weights_filename = 'trained_parameters/drqn_weights/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(eps_start), str(eps_end), str(eps_decay), str(sight), str(use_belief_state), 'drqn']) + '.pth'

    stats = drqn(env, weights_filename, no_episodes, eps_start, eps_end, eps_decay, sight)

    if save_stats:
        stats_directory = 'training_analysis/drqn/' + '_'.join([start_time, str(env.board_height), str(env.board_width), env.reward_type, str(no_episodes), str(eps_start), str(eps_end), str(eps_decay), str(sight), str(use_belief_state)])
        mkdir(stats_directory)
        stats_filename = stats_directory + '/stats.txt'
        save_training_analysis_to_file(stats, stats_filename)
        return weights_filename, stats_directory

    return weights_filename, None
