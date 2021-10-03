import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

sns.set('poster', rc={"xtick.bottom": True, "ytick.left": True,
                      'axes.edgecolor': '.2',
                      "font.weight": 'bold',
                      "axes.titleweight": 'bold',
                      'axes.labelweight': 'bold'})


def create_logger(name, log_dir):
    """
    Creates a logger with a stream handler and file handler.

    Args:
        name (str): The name of the logger.
        log_dir (str): The directory in which to save the logs.

    Returns:
        logger: the logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Set logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    return logger


def dict_to_str(dictionary, level=0):
    """
    A helper function to log dictionaries in a pretty way.

    Args:
        dictionary (dict): A general python dictionary.
        level (int): A recursion level counter, sets the visual indentation.

    Returns:
        str: A text representation for the dictionary.
    """
    message = ''
    for key, value in dictionary.items():
        if isinstance(value, dict):
            message += ' ' * level * 2 + str(key) + ':\n' + dict_to_str(value, level + 1)
        else:
            message += ' ' * level * 2 + str(key) + ': ' + str(value) + '\n'
    return message


def string_representer(dumper, data):
    """
    Add a custom string representer to use block literals for multiline strings.
    """
    if len(data.splitlines()) > 1:
        return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data, style='|')
    return dumper.represent_scalar(tag='tag:yaml.org,2002:str', value=data)


def save_yaml_file(path, content):
    """
    Save a YAML file (usually an input / restart file, but also conformers file)

    Args:
        path (str): The YAML file path to save.
        content (list, dict): The content to save.
    """
    if not isinstance(path, str):
        raise TypeError(f'path must be a string, got {path} which is a {type(path)}')
    yaml.add_representer(str, string_representer)
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)


def plot_train_val_loss(log_file):
    """
    Plots the training and validation loss by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    pattern = '-?[\d.]+(?:e-?\d+)?'
    val_rmse, train_rmse = [], []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'Epoch' in line and 'Overall Validation RMSE' in line and 'Best' not in line:
                # index 0 is epoch, index 1 is RMSE, index 2 is MAE
                val_rmse.append(float(re.findall(pattern, line)[1].rstrip()))
            elif 'Epoch' in line and 'Overall Training RMSE' in line:
                train_rmse.append(float(re.findall(pattern, line)[1].rstrip()))
            elif 'Starting training...' in line:
                break

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(train_rmse))[::-1], train_rmse, label='Train RMSE')
    ax.plot(np.arange(len(val_rmse))[::-1], val_rmse, label='Val RMSE')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    fig.savefig(os.path.join(os.path.dirname(log_file), 'train_val_rmse.pdf'), bbox_inches='tight')


def plot_lr(log_file):
    """
    Plots the learning rate by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    lr = []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'lr_0' in line:
                lr.append(float(line.split(' ')[-1].rstrip()))
            if 'Steps per epoch:' in line:  # only present when using noam
                steps_per_epoch = line.split(' ')[-1].rstrip()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(lr))[::-1], lr)
    ax.set_xlabel(f'Steps (steps per epoch: {steps_per_epoch})')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    fig.savefig(os.path.join(os.path.dirname(log_file), 'learning_rate.pdf'), bbox_inches='tight')


def plot_gnorm_pnorm(log_file):
    """
    Plots the gradient norm and parameter norm by parsing the log file.

    Args:
        log_file (str): The path to the log file created during training.
    """
    gnorm, pnorm = [], []
    with open(log_file) as f:
        lines = f.readlines()
        for line in reversed(lines):
            if 'PNorm' in line:
                # split gives ['Training', 'RMSE:', '0.00327,', 'PNorm:', '127.8966,', 'GNorm:', '2.5143']
                pnorm.append(float(line.split()[4].rstrip(',')))
                gnorm.append(float(line.split()[6].rstrip(',')))
            if 'Steps per epoch:' in line:
                steps_per_epoch = line.split()[-1].rstrip()
                break

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(pnorm))[::-1], pnorm)
    ax.set_xlabel(f'Steps (steps per epoch: {steps_per_epoch})')
    ax.set_ylabel('Parameter Norm')
    fig.savefig(os.path.join(os.path.dirname(log_file), 'pnorm.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(np.arange(len(gnorm))[::-1], gnorm)
    ax.set_xlabel(f'Steps (steps per epoch: {steps_per_epoch})')
    ax.set_ylabel('Gradient Norm')
    ax.set_yscale('log')
    fig.savefig(os.path.join(os.path.dirname(log_file), 'gnorm.pdf'), bbox_inches='tight')
