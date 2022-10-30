from distutils.command.config import config
import logging
from os import mkdir,makedirs
from pathlib import Path
from shutil import copyfile
import os
from tensorflow.keras import mixed_precision
#from trainers.trainer import Trainer
import tensorflow as tf
import numpy as np
import random
import argparse

def custom_parser():
    """create console arg parser
    """
    parse = argparse.ArgumentParser(prog = 'argparseDemo',description='the message info before help info',
    epilog="the message info after help info")
    modes=["train","predict","enhance"]
    parse.add_argument('-m', '--mode', choices=modes,type=str,help='set execute mode : train or predict',default=None)
    parse.add_argument('-c', '--config', help='set model config used',default=None)
    parse.add_argument('--version',action = 'version',version = 'just demo ~~')
    return parse


def enhance_parser():
    """create console arg parser
    TODO
    """
    parse = argparse.ArgumentParser(prog = 'argparseDemo',description='the message info before help info',
    epilog="the message info after help info")
    modes=["train","predict","enhance"]
    parse.add_argument('-m', '--mode', choices=modes,type=str,help='set execute mode : train or predict',default=None)
    parse.add_argument('-c', '--config', help='set model config used',default=None)
    parse.add_argument('--version',action = 'version',version = 'just demo ~~')
    return parse



def show_parser(args):
    print('=======args=======')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('=======args=======\n')
    