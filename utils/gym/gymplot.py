import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython import display

def plot_init(env):
    plt.figure(figsize=(9,9),dpi=300)
    return plt.imshow(env.render(mode='rgb_array')) # only call this once

def plot_next(img, env):
    img.set_data(env.render(mode='rgb_array')) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    return img