import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import copy
# import scipy as sp
# import matplotlib as mpl
# import matplotlib.image as image
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import imageio
# from os import listdir, unlink
# from os.path import isfile, join
# import xarray as xr
# import gym

from lib.utils import *


def generate_transition_probabilities(board_height, board_width, current_state, action_index, sight = 2, mouse_pos_dist = True):
    cat_state = current_state[:board_height * board_width]
    mouse_state = current_state[board_height * board_width:]
    cat_pos_index = list(cat_state).index(1)
    cat_pos = board_pos_index_to_board_pos(cat_pos_index, board_height, board_width)
    action = action_index_to_action(action_index)
    
