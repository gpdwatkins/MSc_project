import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from os.path import join
from imageio import get_writer, imread
from lib.utils import *
from datetime import datetime
from qlearning.qlearning import *
from deepqlearning.dqn_agent import DQNAgent
from drqn.drqn_agent import DRQNAgent
import torch

# This file contains various functions to produce graphics

def imscatter(x, y, image, ax=None, zoom=1):
    # Used to generate a scatter plot on 2d axes where the points are user-defined images

    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_fig = True, walls = None):
    # Generates and saves an image of the current board configuration
    # We can specify whether we want the figure to be shown or saved silently
    # These images are knitted together to form the gif
    # This function is only called by the play_cat_and_mouse function
    # We can also specify (interior) wall segments on our board using the keyword arg walls
    # The walls argument should be a list of 'coordinates' specifying the wall midpoint
    # It has the (row_midpoint, col_midpoint)
    # So the wall segment between the square in row 1, col 3 and the square in row 2, col 3
    # would be indicated using the argument (1.5, 3)
    # We can interpret this as 'the wall segment between rows 1 and 2, across column 3'

    fig_width = 20
    fig = plt.figure(figsize=(fig_width, 0.8 * fig_width * board_height/board_width))
    ax1 = fig.add_subplot(111)

    sns.heatmap(mouse_pos_prob_dist, cmap="Blues", vmin=0, vmax=1, linewidths=.5, ax = ax1, cbar_kws={'label': "Cat's Probability Distribution of Mouse Position", })
    ax1.figure.axes[-1].yaxis.label.set_size(16)
    for item in ax1.figure.axes[-1].get_yticklabels():
        item.set_fontsize(14)

    imscatter(cat_pos[1] + 1/2, cat_pos[0] + 1/2, plt.imread('images/tom.gif'), zoom = 0.1 * fig_width/10 * 6/board_width, ax=ax1)
    imscatter(mouse_pos[1] + 1/2, mouse_pos[0] + 1/2, plt.imread('images/jerry.gif'), zoom = (2/3) * 0.1 * fig_width/10 * 6/board_width, ax=ax1)

    ax1.plot([0, 0], [0, board_height], color='black', linewidth=3*(6/board_width)**(1/2))
    ax1.plot([0, board_width], [0, 0], color='black', linewidth=3*(6/board_width)**(1/2))
    ax1.plot([0, board_width], [board_height - .005, board_height - .005], color='black', linewidth=3*(6/board_width)**(1/2))
    ax1.plot([board_width - .005, board_width - .005], [0, board_height], color='black', linewidth=3*(6/board_width)**(1/2))

    if not walls == None:
        for wall_midpoint in walls:
            midpoint_x_coord, midpoint_y_coord = wall_midpoint_to_coords(wall_midpoint)
            if midpoint_x_coord%1 == 0 and (midpoint_y_coord - 0.5)%1 == 0:
                ax1.plot([midpoint_x_coord, midpoint_x_coord], [midpoint_y_coord - 0.5, midpoint_y_coord + 0.5], color='black', linewidth=3*(6/board_width)**(1/2))
            elif (midpoint_x_coord - 0.5)%1 == 0 and midpoint_y_coord%1 == 0:
                ax1.plot([midpoint_x_coord - 0.5, midpoint_x_coord + 0.5], [midpoint_y_coord, midpoint_y_coord], color='black', linewidth=3*(6/board_width)**(1/2))
            else:
                raise Exception('Wall segments should be specified using their midpoint')

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plt.gca().invert_yaxis()

    plt.savefig(join('graphics_gif', filename))

    if show_fig:
        plt.show()


def generate_gif(image_filenames, gif_filename):
    # Knits together the images (whose filenames are given as inputs)
    # to create a gif showing a full episode of cat-and-mouse

    with get_writer(gif_filename, mode='I', duration=0.5, loop=1) as writer:
        for filename in image_filenames:
            image = imread(join('graphics_gif/', filename))
            writer.append_data(image)


def show_policy(mouse_pos, board_height, board_width, parameter_filename = None, walls = None):
    # Given a parameter_filename and a mouse_pos, shows what action the cat would take
    # if it were located in each of the other squares on the board
    # parameter_filename is a string representing a filename -
    # either a .txt file (which means it was trained using q learning)
    # or a .pth file (which means it was trained using a deep neural network)
    # If no parameter_filename is specified, a random policy is shown

    action_to_arrow_files = { \
    0:'images/icon_NW.gif', \
    1:'images/icon_N.gif', \
    2:'images/icon_NE.gif', \
    3:'images/icon_W.gif', \
    4:'images/icon_X.gif', \
    5:'images/icon_E.gif', \
    6:'images/icon_SW.gif', \
    7:'images/icon_S.gif', \
    8:'images/icon_SE.gif', \
    }

    fig_width = 20
    fig = plt.figure(figsize=(fig_width, fig_width * board_height/board_width))
    ax1 = fig.add_subplot(111)

    mouse_pos_prob_dist = np.zeros([board_height, board_width])

    sns.heatmap(mouse_pos_prob_dist, cmap="Blues", vmin=0, vmax=1, linewidths=.5, ax = ax1, cbar = None)
    ax1.figure.axes[-1].yaxis.label.set_size(12)

    imscatter(mouse_pos[1] + 1/2, mouse_pos[0] + 1/2, plt.imread('images/jerry.gif'), zoom = 0.1 * fig_width/10 * 6/board_width, ax=ax1)

    ax1.plot([0, 0], [0, board_height], color='black', linewidth=3*(6/board_width)**(1/2))
    ax1.plot([0, board_width], [0, 0], color='black', linewidth=3*(6/board_width)**(1/2))
    ax1.plot([0, board_width], [board_height - .005, board_height - .005], color='black', linewidth=3*(6/board_width)**(1/2))
    ax1.plot([board_width - .005, board_width - .005], [0, board_height], color='black', linewidth=3*(6/board_width)**(1/2))

    if not walls == None:
        for wall_midpoint in walls:
            midpoint_x_coord, midpoint_y_coord = wall_midpoint_to_coords(wall_midpoint)
            if midpoint_x_coord%1 == 0 and (midpoint_y_coord - 0.5)%1 == 0:
                ax1.plot([midpoint_x_coord, midpoint_x_coord], [midpoint_y_coord - 0.5, midpoint_y_coord + 0.5], color='black', linewidth=3*(6/board_width)**(1/2))
            elif (midpoint_x_coord - 0.5)%1 == 0 and midpoint_y_coord%1 == 0:
                ax1.plot([midpoint_x_coord - 0.5, midpoint_x_coord + 0.5], [midpoint_y_coord, midpoint_y_coord], color='black', linewidth=3*(6/board_width)**(1/2))
            else:
                raise Exception('Wall segments should be specified using their midpoint')


    cat_pos_actions = {}
    if parameter_filename == None:
        print("No parameter_filename provided; using random policy")
        for cat_vert_pos in range(board_height):
            for cat_horz_pos in range(board_width):
                cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = np.random.choice(list(action_to_arrow_files.keys()))
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.txt':
        metadata = extract_training_metadata(parameter_filename)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Parameters were generated using different board size')
        if parameter_filename.find('qlearning_qvalues/') == -1:
            parameter_filename = 'qlearning_qvalues/' + parameter_filename
            if parameter_filename.find('trained_parameters/') == -1:
                parameter_filename = 'trained_parameters/' + parameter_filename
        qvalues_dict = load_qvalues_from_file(parameter_filename)
        policy_dict = get_greedy_policy(qvalues_dict)
        for cat_vert_pos in range(board_height):
            for cat_horz_pos in range(board_width):
                cat_pos = (cat_vert_pos, cat_horz_pos)
                state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
                cat_pos_actions[cat_pos] = policy_dict[state_index]

        # for (state_index, action) in policy.items():
        #     state_cat_pos, state_mouse_pos = state_index_to_positions(state_index, board_height, board_width)
        #     if mouse_pos == state_mouse_pos:
        #         cat_pos_actions[state_cat_pos] = action
    elif type(parameter_filename) is str and parameter_filename[-4:] == '.pth':
        metadata = extract_training_metadata(parameter_filename)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Parameters were generated using different board size')
        if metadata['algorithm'] == 'dqn' and parameter_filename.find('dqn_weights/') == -1:
            parameter_filename = 'dqn_weights/' + parameter_filename
        if metadata['algorithm'] == 'drqn' and parameter_filename.find('drqn_weights/') == -1:
            parameter_filename = 'drqn_weights/' + parameter_filename
        if parameter_filename.find('trained_parameters/') == -1:
            parameter_filename = 'trained_parameters/' + parameter_filename
        if metadata['algorithm'] == 'dqn':
            agent = DQNAgent(state_size = 2 * board_height * board_width, action_size = len(list(action_to_arrow_files.keys())), seed = 0)
            agent.qnetwork_behaviour.load_state_dict(torch.load(parameter_filename))
            for cat_vert_pos in range(board_height):
                for cat_horz_pos in range(board_width):
                    nn_state = positions_to_nn_input((cat_vert_pos, cat_horz_pos), mouse_pos, board_height, board_width)
                    cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = agent.act(nn_state)
        elif metadata['algorithm'] == 'drqn':
            agent = DRQNAgent(state_size = 2 * board_height * board_width, action_size = len(list(action_to_arrow_files.keys())), seed = 0)
            agent.drqn_behaviour.load_state_dict(torch.load(parameter_filename))
            for cat_vert_pos in range(board_height):
                for cat_horz_pos in range(board_width):
                    nn_state = positions_to_nn_input((cat_vert_pos, cat_horz_pos), mouse_pos, board_height, board_width)
                    drqn_state = nn_input_as_drqn_input(board_height, board_width, nn_state)
                    hidden_state = agent.init_hidden()
                    cat_pos_actions[(cat_vert_pos, cat_horz_pos)], hidden = agent.act(drqn_state, hidden_state)

    else:
        raise ValueError('parameter_filename type not recognised. Should be None, or .txt or .pth filename')

    for (state_cat_pos, action) in cat_pos_actions.items():
        if not state_cat_pos == mouse_pos:
            imscatter(state_cat_pos[1] + 1/2, state_cat_pos[0] + 1/2, plt.imread(action_to_arrow_files[action]), zoom = 0.1 * fig_width/10, ax=ax1)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.xlim(0,board_width)
    plt.ylim(0,board_height)
    plt.gca().invert_yaxis()

    plt.show()


def plot_episode_stats(stats, smoothing_window=100, show_fig=True):
    # generates plots showing various statistics related to training
    # including episode length, episode reward and episode number against time
    # directory_name = 'training_analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    # mkdir(join('graphics_training_analysis', directory_name))

    # ax1 = fig1.add_subplot(111)
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.saved_episodes, stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    # plt.savefig(join('graphics_training_analysis', directory_name, 'episode_lengths'))
    if show_fig:
        plt.show(fig1)
    else:
        plt.close(fig1)

    # ax2 = fig1.add_subplot(211)
    # Plot the (smoothed) episode length over time
    fig2 = plt.figure(figsize=(10,5))
    episode_lengths_smoothed = pd.Series(stats.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(stats.saved_episodes, episode_lengths_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Smoothed)")
    plt.title("Episode Length over Time (Smoothed over window size {})".format(smoothing_window))
    # plt.savefig(join('graphics_training_analysis', directory_name, 'episode_lengths_smoothed'))
    if show_fig:
        plt.show(fig2)
    else:
        plt.close(fig2)

    # ax3 = fig1.add_subplot(311)
    # Plot the episode reward over time
    fig3 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(stats.saved_episodes, rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    # plt.savefig(join('graphics_training_analysis', directory_name, 'episode_rewards_smoothed'))
    if show_fig:
        plt.show(fig3)
    else:
        plt.close(fig3)

    # ax4 = fig1.add_subplot(411)
    # Plot time steps and episode number
    fig4 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    # plt.savefig(join('graphics_training_analysis', directory_name, 'episodes_per_time_step'))
    if show_fig:
        plt.show(fig4)
    else:
        plt.close(fig4)

    # plt.savefig(join('graphics_training_analysis', 'training_analysis_' + datetime.now().strftime('%Y%m%d_%H%M') + '.png'))

    return fig1, fig2, fig3, fig4


def compare_training_graphics(list_of_filepaths, smoothing_window = 100):
    # takes a list of stats filepaths as input and overlays them on the same axes

    episode_lengths = {}
    episode_rewards = {}

    i = 1
    for filepath in list_of_filepaths:
        metadata = extract_analysis_metadata(filepath)
        stats = load_training_analysis_from_file(filepath)
        episode_lengths[metadata['training_algorithm'] + str(i)] = stats.episode_lengths
        episode_rewards[metadata['training_algorithm'] + str(i)] = stats.episode_rewards
        i+=1
        # episode_lengths.append(stats.episode_lengths)
        # episode_rewards.append(stats.episode_rewards)

    fig1 = plt.figure(figsize=(10,5))
    # legend_labels1 = []
    for episode_length_label, episode_length_data in episode_lengths.items():
        episode_lengths_smoothed = pd.Series(episode_length_data).rolling(smoothing_window, min_periods=smoothing_window).mean()
        # legend_labels1.append(episode_length_label)
        plt.plot(episode_lengths_smoothed, label = episode_length_label)
    plt.title("Episode Length over Time (Smoothed over window size {})".format(smoothing_window))
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Smoothed)")
    plt.legend()


    fig2 = plt.figure(figsize=(10,5))
    # legend_labels2 = []
    for episode_reward_label, episode_reward_data in episode_rewards.items():
        episode_rewards_smoothed = pd.Series(episode_reward_data).rolling(smoothing_window, min_periods=smoothing_window).mean()
        # legend_labels2.append(episode_length_label)
        plt.plot(episode_rewards_smoothed, label = episode_reward_label)
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.legend()
