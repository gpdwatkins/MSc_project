import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from imageio import get_writer, imread
from os import listdir, unlink
from os.path import isfile, join
from lib.utils import initialise_board, initialise_cat_mouse_positions, initialise_mouse_prob_dist, cat_can_see_mouse
from datetime import datetime

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_fig = True):
    fig_width = 10
    fig = plt.figure(figsize=(fig_width, 0.8 * fig_width * board_height/board_width))
    ax1 = fig.add_subplot(111)

    sns.heatmap(mouse_pos_prob_dist, cmap="Blues", vmin=0, vmax=1, linewidths=.5, ax = ax1, cbar_kws={'label': "Cat's Probability Distribution of Mouse Position", })
    ax1.figure.axes[-1].yaxis.label.set_size(12)

    imscatter(cat_pos[1] + 1/2, cat_pos[0] + 1/2, plt.imread('tom.gif'), zoom = 0.1 * fig_width/10, ax=ax1)
    imscatter(mouse_pos[1] + 1/2, mouse_pos[0] + 1/2, plt.imread('jerry.gif'), zoom = (2/3) * 0.1 * fig_width/10, ax=ax1)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    plt.savefig(join('gif_graphics', filename))

    if show_fig:
        plt.show()


def generate_gif(image_filenames, gif_filename):
    with get_writer(gif_filename, mode='I', duration=0.5, loop=1) as writer:
        for filename in image_filenames:
            image = imread(join('gif_graphics/', filename))
            writer.append_data(image)


def play_cat_and_mouse(board_height, board_width, show_figs = True):
    # assumimg cat has sight 2 in each direction (i.e. can see a 5x5 grid around iteself)
    # cat and mouse move uniformly (can move any direction, or stay put, with prob 1/9)
    # cat policy doesn't update - stays uniform
    # note that if either cat or mouse attempts to move into the wall (even diagonally) they stay where they are

    start_time = datetime.now().strftime('%Y%m%d_%H%M')

    for file in listdir('gif_graphics'):
        file_with_path = join('gif_graphics', file)
        if isfile(file_with_path):
            unlink(file_with_path)

    # for file in listdir('gif_graphics/the_gif'):
    #     file_with_path = join('gif_graphics/the_gif', file)
    #     if isfile(file_with_path):
    #         unlink(file_with_path)

    filenames = []

    # Initialise stuff
    initial_board = np.array(initialise_board(board_height, board_width))

    cat_pos, mouse_pos = initialise_cat_mouse_positions(board_height, board_width)


    board = np.array(np.zeros([board_height, board_width]), dtype='O')
    board[cat_pos] = 'C'
    board[mouse_pos] = 'M'

    mouse_pos_prob_dist = initialise_mouse_prob_dist(board_height, board_width, cat_pos, mouse_pos)

    print('Starting position')

    filename = filename = 'gif_graphic_' + start_time + '_0' + '.png'
    filenames.append(filename)
    generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)

    iter = 1
    while cat_pos != mouse_pos:
        cat_vert_move = np.random.choice((-1,0,1))
        cat_horz_move = np.random.choice((-1,0,1))
        mouse_vert_move = np.random.choice((-1,0,1))
        mouse_horz_move = np.random.choice((-1,0,1))
        new_cat_pos = (cat_pos[0] + cat_vert_move, cat_pos[1] + cat_horz_move)
        if (new_cat_pos[0] in range(board_height) and new_cat_pos[1] in range(board_width)):
            cat_pos = new_cat_pos
        new_mouse_pos = (mouse_pos[0] + mouse_vert_move, mouse_pos[1] + mouse_horz_move)
        if (new_mouse_pos[0] in range(board_height) and new_mouse_pos[1] in range(board_width)):
            mouse_pos = new_mouse_pos
        board = np.array(np.zeros([board_height, board_width]), dtype='O')
        board[cat_pos] = 'C'
        board[mouse_pos] = 'M'

        if cat_can_see_mouse(cat_pos, mouse_pos):
            mouse_pos_prob_dist = np.zeros((board_height, board_width))
            mouse_pos_prob_dist[mouse_pos] = 1
        else:
            new_mouse_pos_prob_dist = np.zeros((board_height, board_width))
            for row in range(board_height):
                for col in range(board_width):
                    for row_offset in [-1, 0, 1]:
                        for col_offset in [-1, 0, 1]:
                            if (((row + row_offset) in range(board_height)) and ((col + col_offset) in range(board_width))):
                                if not ((abs(row + row_offset - cat_pos[0]) <= 2) and (abs(col + col_offset - cat_pos[1]) <= 2)):
                                    new_mouse_pos_prob_dist[row + row_offset, col + col_offset] += (1/9) * mouse_pos_prob_dist[row,col]
                            else:
                                if not ((abs(row - cat_pos[0]) <= 2) and (abs(col - cat_pos[1]) <= 2)):
                                    new_mouse_pos_prob_dist[row, col] += (1/9) * mouse_pos_prob_dist[row,col]

            sum_mouse_pos_prob_dist = np.sum(new_mouse_pos_prob_dist)
            new_mouse_pos_prob_dist /= sum_mouse_pos_prob_dist
            mouse_pos_prob_dist = deepcopy(new_mouse_pos_prob_dist)



        print('\n Iteration %d' % iter)

        filename = 'gif_graphic_' + start_time + '_' + str(iter) + '.png'
        filenames.append(filename)

        gif_filename = 'gif_graphics/the_gif/output_' + start_time + '.gif'

        generate_fig(cat_pos, mouse_pos, mouse_pos_prob_dist, board_height, board_width, filename, show_figs)
        iter += 1

    generate_gif(filenames, gif_filename)


play_cat_and_mouse(4,6)
