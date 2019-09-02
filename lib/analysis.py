import numpy as np
from datetime import datetime
from collections import namedtuple
import itertools
from os.path import join
# from os import mkdir
from lib.utils import *
from lib.generate_graphics import plot_episode_stats
from qlearning.qlearning import *
from deepqlearning.dqn_agent import DQNAgent
from drqn.drqn_agent import DRQNAgent
import torch

def training_graphical_analysis(stats_directory, smoothing_window=1, show_figs = True, save_figs = True):

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
                return int(stats.saved_episodes[i]), round(np.mean(stats.episode_lengths[i:]),2)
            else:
                failed_on = 'variance'
        else:
            failed_on = 'mean'
        i += averaging_window
    raise Exception('Insufficient episodes for %s of episode lengths to stabilise' % failed_on)


def test_policy(env, board_height, board_width, no_episodes, parameter_filename = None, seed = None):
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    if not seed is None:
        np.random.seed(seed)

    if parameter_filename == None:
        print("No parameter_filename provided; using random policy")
        policy_type = 'random'
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.txt':
        policy_type = 'state-qvalues_dict'
        metadata = extract_training_metadata(parameter_filename)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Parameters were generated using different board size')
        if parameter_filename.find('qlearning_qvalues/') == -1:
            parameter_filename = 'qlearning_qvalues/' + parameter_filename
            if parameter_filename.find('trained_parameters/') == -1:
                parameter_filename = 'trained_parameters/' + parameter_filename
        qvalues_dict = load_qvalues_from_file(parameter_filename)
        policy_dict = get_greedy_policy(qvalues_dict)
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.pth':
        metadata = extract_training_metadata(parameter_filename)
        if metadata['algorithm'] == 'dqn':
            policy_type = 'dqn_weights'
        elif metadata['algorithm'] == 'drqn':
            policy_type = 'drqn_weights'
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Parameters were generated using different board size')
        if metadata['algorithm'] == 'dqn' and parameter_filename.find('dqn_weights/') == -1:
            parameter_filename = 'dqn_weights/' + parameter_filename
        if metadata['algorithm'] == 'drqn' and parameter_filename.find('drqn_weights/') == -1:
            parameter_filename = 'drqn_weights/' + parameter_filename
        if parameter_filename.find('trained_parameters/') == -1:
            parameter_filename = 'trained_parameters/' + parameter_filename
        if policy_type == 'dqn_weights':
            agent = DQNAgent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
            agent.qnetwork_behaviour.load_state_dict(torch.load(parameter_filename))
        elif policy_type == 'drqn_weights':
            agent = DRQNAgent(state_size = 2 * board_height * board_width, action_size = 9, seed = 0)
            agent.drqn_behaviour.load_state_dict(torch.load(parameter_filename))
    else:
        raise ValueError('parameter_filename type not recognised. Should be None, dict or .pth filename')

    stats = namedtuple("Stats",["episode_lengths", "episode_rewards"])(
        episode_lengths=np.zeros(no_episodes),
        episode_rewards=np.zeros(no_episodes))

    for episode in range(1, no_episodes+1):

        current_state_index = env.reset()

        for timestep in itertools.count():
            if policy_type == 'random':
                action_index = np.random.randint(9)
            elif policy_type == 'state-qvalues_dict':
                action_index = policy_dict[current_state_index]
            elif policy_type == 'dqn_weights':
                cat_pos, mouse_pos = state_index_to_positions(current_state_index, board_height, board_width)
                nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
                action_index = agent.act(nn_state)
            elif policy_type == 'drqn_weights':
                cat_pos, mouse_pos = state_index_to_positions(current_state_index, board_height, board_width)
                nn_state = positions_to_nn_input(cat_pos, mouse_pos, board_height, board_width)
                drqn_state = nn_input_as_drqn_input(board_height, board_width, nn_state)
                if timestep == 0:
                    hidden = agent.init_hidden()
                action_index, hidden = agent.act(drqn_state, hidden)
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
