

import random
import numpy as np


def random_color_i():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def random_color_v(v):
    
    rgbl=[v%255,0,0]
    random.shuffle(rgbl)
    return np.array(rgbl)/255

def random_hex():
    rand_colors = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
    return rand_colors

def random_color_f():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return np.array(rgbl)/255

import seaborn as sns
def random_sns_pallete():
    palette = sns.color_palette(None, 3)
    return palette

def random_color_list(n):
    return [random_hex() for i in range(n)]

def random_color_array(n):
    color_arr=np.zeros((n,3))
    for i in range(n):
        color_arr[i,:]=random_color_f()
    return color_arr
