import numpy as np
from datetime import datetime
from collections import namedtuple
import itertools
from os.path import join
# from os import mkdir
from lib.utils import *
from lib.generate_graphics import plot_episode_stats
from deepqlearning.dqn_agent import Agent
import torch

def training_graphical_analysis(stats_directory, smoothing_window=100, show_figs = True, save_figs = True):

    # start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # directory_name = 'training_analysis/' + '_'.join([start_time, 'smoothingwindow', str(smoothing_window)])
    # mkdir(directory_name)

    stats = load_training_analysis_from_file(stats_directory + '/stats.txt')

    fig1, fig2, fig3, fig4 = plot_episode_stats(stats, smoothing_window = smoothing_window, show_fig = show_figs)

    if save_figs:
        fig1.savefig(join(stats_directory, 'episode_lengths'))
        fig2.savefig(join(stats_directory, 'episode_lengths_smoothed_' + str(smoothing_window)))
        fig3.savefig(join(stats_directory, 'episode_rewards_smoothed_' + str(smoothing_window)))
        fig4.savefig(join(stats_directory, 'episodes_per_time_step'))



    # return directory_name

def stabilisation_analysis(stats_directory, averaging_window = 100, mean_tolerance = 5, var_tolerance = 25):
    # Checks whether the mean and variance of episode lengths has stabilised
    # We consider the episode lengths to have stabilised at episode i if:
    # The difference between the mean over the next (averaging_window) episodes
    # and the mean over (all) the episodes after that
    # is less than mean_tolerance
    # and the variance over all episodes from i onwards is less than var_tolerance

    stats = load_training_analysis_from_file(stats_directory)

    i = 0
    failed_on = ''
    while i + 2 * averaging_window - 1 <= len(stats[0]):
        average_window_1 = np.mean(stats.episode_lengths[i:i + averaging_window])
        # average_window_2 = np.mean(stats.episode_lengths[i + averaging_window:i + 2 * averaging_window - 1])
        average_window_2 = np.mean(stats.episode_lengths[i + averaging_window:])
        if (abs(average_window_1 - average_window_2) < mean_tolerance):
            variance_remaining_episodes = np.var(stats.episode_lengths[i:])
            if variance_remaining_episodes < var_tolerance:
                return i, round(np.mean(stats.episode_lengths[i:]),2)
            else:
                failed_on = 'variance'
        else:
            failed_on = 'mean'
        i += averaging_window
    raise Exception('Insufficient episodes for %s of episode lengths to stabilise' % failed_on)


def test_policy(env, board_height, board_width, no_episodes, policy = None, seed = None):
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    if not seed is None:
        np.random.seed(seed)

    if policy == None:
        policy_type = 'random'
    elif type(policy) is str and policy[-4:] == '.txt':
        policy_type = 'state-action_dict'
        metadata = extract_training_metadata(policy)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        policy_dict = load_policy_from_file(policy)
    elif type(policy) is str and policy[-4:] == '.pth':
        policy_type = 'nn_weights'
        metadata = extract_training_metadata(policy)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        agent = Agent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
        agent.qnetwork_behaviour.load_state_dict(torch.load(policy))
    else:
        raise ValueError('Policy type not recognised. Should be None, dict or .pth filename')

    stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
        episode_lengths=np.zeros(no_episodes),
        episode_rewards=np.zeros(no_episodes))

    for episode in range(1, no_episodes+1):

        current_state_index = env.reset()

        for timestep in itertools.count():
            if policy_type == 'random':
                action_index = np.random.randint(9)
            elif policy_type == 'state-action_dict':
                action_index = policy_dict[current_state_index]
            elif policy_type == 'nn_weights':
                cat_pos, mouse_pos = state_index_to_positions(current_state_index, board_height, board_width)
                nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
                action_index = agent.act(nn_state)
            else:
                raise ValueError('Never should have reached this point!')

            # Take a step
            next_state, reward, done, _ = env.step(action_index)

            # Update statistics
            stats.episode_rewards[episode-1] += reward
            if done:
                stats.episode_lengths[episode-1] = timestep
                break

            current_state_index = next_state

    return np.average(stats.episode_lengths), np.average(stats.episode_rewards)
