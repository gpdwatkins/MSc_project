import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from os.path import join
from os import mkdir
from imageio import get_writer, imread
from lib.utils import *
from datetime import datetime
from deepqlearning.dqn_agent import Agent
import torch

# this file contains various functions to produce graphics

def imscatter(x, y, image, ax=None, zoom=1):
    # used for a scatter plot on 2d axes
    # where the points are user-defined images
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    # artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), frameon=False)
        # artists.append(ax.add_artist(ab))
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    # return artists


def generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_fig = True):
    # generates and saves an image of the current board configuration
    # we can specify whether we want the figure to be shown or saved silently
    # these images are knitted together to form the gif
    fig_width = 20
    fig = plt.figure(figsize=(fig_width, 0.8 * fig_width * board_height/board_width))
    ax1 = fig.add_subplot(111)

    sns.heatmap(mouse_pos_prob_dist, cmap="Blues", vmin=0, vmax=1, linewidths=.5, ax = ax1, cbar_kws={'label': "Cat's Probability Distribution of Mouse Position", })
    ax1.figure.axes[-1].yaxis.label.set_size(16)
    for item in ax1.figure.axes[-1].get_yticklabels():
        item.set_fontsize(14)

    imscatter(cat_pos[1] + 1/2, cat_pos[0] + 1/2, plt.imread('images/tom.gif'), zoom = 0.1 * fig_width/10 * 6/board_width, ax=ax1)
    imscatter(mouse_pos[1] + 1/2, mouse_pos[0] + 1/2, plt.imread('images/jerry.gif'), zoom = (2/3) * 0.1 * fig_width/10 * 6/board_width, ax=ax1)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    # plt.gca().invert_yaxis()

    plt.savefig(join('graphics_gif', filename))

    if show_fig:
        plt.show()


def generate_gif(image_filenames, gif_filename):
    # knits together the images (whose filenames are given as inputs)
    # to create a gif showing a full episode of cat-and-mouse
    with get_writer(gif_filename, mode='I', duration=0.5, loop=1) as writer:
        for filename in image_filenames:
            image = imread(join('graphics_gif/', filename))
            writer.append_data(image)

#
# def show_policy(mouse_pos, board_height, board_width, policy = None):
#     # given a policy and a mouse_pos, shows what action the cat would take
#     # if it were located in each of the other squares on the board
#     # if no policy is specified, a random policy is shown
#     # policy could alternatively be a dict mapping state_index to action
#     # or a string specifying a file path to some trained weights
#     action_to_arrow_files = { \
#     0:'icon_NW.gif', \
#     1:'icon_N.gif', \
#     2:'icon_NE.gif', \
#     3:'icon_W.gif', \
#     4:'icon_X.gif', \
#     5:'icon_E.gif', \
#     6:'icon_SW.gif', \
#     7:'icon_S.gif', \
#     8:'icon_SE.gif', \
#     }
#
#     fig_width = 20
#     fig = plt.figure(figsize=(fig_width, fig_width * board_height/board_width))
#     ax1 = fig.add_subplot(111)
#
#     mouse_pos_prob_dist = np.zeros([board_height, board_width])
#
#     sns.heatmap(mouse_pos_prob_dist, cmap="Blues", vmin=0, vmax=1, linewidths=.5, ax = ax1, cbar = None)
#     ax1.figure.axes[-1].yaxis.label.set_size(12)
#
#     imscatter(mouse_pos[1] + 1/2, mouse_pos[0] + 1/2, plt.imread('jerry.gif'), zoom = 0.1 * fig_width/10 * 6/board_width, ax=ax1)
#
#     cat_pos_actions = {}
#     if policy == None:
#         print("No policy provided; using random policy")
#         for cat_vert_pos in range(board_height):
#             for cat_horz_pos in range(board_width):
#                 cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = np.random.choice(list(action_to_arrow_files.keys()))
#     elif type(policy) is dict:
#         for (state_index, action) in policy.items():
#             state_cat_pos, state_mouse_pos = state_index_to_positions(state_index, board_height, board_width)
#             if mouse_pos == state_mouse_pos:
#                 cat_pos_actions[state_cat_pos] = action
#     elif type(policy) is str and policy[-4:] == '.pth':
#         agent = Agent(state_size = 2 + board_height * board_width, action_size = len(list(action_to_arrow_files.keys())), seed = 0)
#         agent.qnetwork_behaviour.load_state_dict(torch.load('checkpoint.pth'))
#         for cat_vert_pos in range(board_height):
#             for cat_horz_pos in range(board_width):
#                 nn_state = positions_to_nn_input((cat_vert_pos, cat_horz_pos), mouse_pos, board_height, board_width)
#                 cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = agent.act(nn_state)
#     else:
#         raise ValueError('Policy type not recognised. Should be None, dict or .pth filename')
#
#     for (state_cat_pos, action) in cat_pos_actions.items():
#         if not state_cat_pos == mouse_pos:
#             imscatter(state_cat_pos[1] + 1/2, state_cat_pos[0] + 1/2, plt.imread(action_to_arrow_files[action]), zoom = 0.1 * fig_width/10, ax=ax1)
#
#     plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
#     plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
#     plt.xlim(0,board_width)
#     plt.ylim(0,board_height)
#     plt.gca().invert_yaxis()
#
#     plt.show()


def show_policy(mouse_pos, board_height, board_width, policy = None):
    # given a policy and a mouse_pos, shows what action the cat would take
    # if it were located in each of the other squares on the board
    # if no policy is specified, a random policy is shown
    # policy could alternatively be a dict mapping state_index to action
    # or a string specifying a file path to some trained weights
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

    cat_pos_actions = {}
    if policy == None:
        print("No policy provided; using random policy")
        for cat_vert_pos in range(board_height):
            for cat_horz_pos in range(board_width):
                cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = np.random.choice(list(action_to_arrow_files.keys()))
    elif type(policy) is dict:
        for cat_vert_pos in range(board_height):
            for cat_horz_pos in range(board_width):
                cat_pos = (cat_vert_pos, cat_horz_pos)
                state_index = positions_to_state_index(cat_pos, mouse_pos, board_height, board_width)
                cat_pos_actions[cat_pos] = policy[state_index]

        # for (state_index, action) in policy.items():
        #     state_cat_pos, state_mouse_pos = state_index_to_positions(state_index, board_height, board_width)
        #     if mouse_pos == state_mouse_pos:
        #         cat_pos_actions[state_cat_pos] = action
    elif type(policy) is str and policy[-4:] == '.pth':
        metadata = extract_training_metadata(policy)
        if not (int(metadata['board_height']) == board_height and int(metadata['board_width']) == board_width):
            raise Exception('Policy was generated using different board size')
        agent = Agent(state_size = 2 + board_height * board_width, action_size = len(list(action_to_arrow_files.keys())), seed = 0)
        if not policy[:16] == 'trained_weights/':
            policy = 'trained_weights/' + policy
        agent.qnetwork_behaviour.load_state_dict(torch.load(policy))
        for cat_vert_pos in range(board_height):
            for cat_horz_pos in range(board_width):
                nn_state = positions_to_nn_input((cat_vert_pos, cat_horz_pos), mouse_pos, board_height, board_width)
                cat_pos_actions[(cat_vert_pos, cat_horz_pos)] = agent.act(nn_state)
    else:
        raise ValueError('Policy type not recognised. Should be None, dict or .pth filename')

    for (state_cat_pos, action) in cat_pos_actions.items():
        if not state_cat_pos == mouse_pos:
            imscatter(state_cat_pos[1] + 1/2, state_cat_pos[0] + 1/2, plt.imread(action_to_arrow_files[action]), zoom = 0.1 * fig_width/10, ax=ax1)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.xlim(0,board_width)
    plt.ylim(0,board_height)
    plt.gca().invert_yaxis()

    plt.show()


def plot_episode_stats(stats, smoothing_window=10, show_fig=True):
    # generates plots showing various statistics related to training
    # including episode length, episode reward and episode number against time
    directory_name = 'training_analysis_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    mkdir(join('graphics_training_analysis', directory_name))

    # ax1 = fig1.add_subplot(111)
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.savefig(join('graphics_training_analysis', directory_name, 'episode_lengths'))
    if show_fig:
        plt.show(fig1)
    else:
        plt.close(fig1)

    # ax2 = fig1.add_subplot(211)
    # Plot the (smoothed) episode length over time
    fig2 = plt.figure(figsize=(10,5))
    episode_lengths_smoothed = pd.Series(stats.episode_lengths).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(episode_lengths_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length (Smoothed)")
    plt.title("Episode Length over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(join('graphics_training_analysis', directory_name, 'episode_lengths_smoothed'))
    if show_fig:
        plt.show(fig2)
    else:
        plt.close(fig2)

    # ax3 = fig1.add_subplot(311)
    # Plot the episode reward over time
    fig3 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(join('graphics_training_analysis', directory_name, 'episode_rewards_smoothed'))
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
    plt.savefig(join('graphics_training_analysis', directory_name, 'episodes_per_time_step'))
    if show_fig:
        plt.show(fig4)
    else:
        plt.close(fig4)

    # plt.savefig(join('graphics_training_analysis', 'training_analysis_' + datetime.now().strftime('%Y%m%d_%H%M') + '.png'))

    # return fig1, fig2, fig3, fig4
    # return fig1
