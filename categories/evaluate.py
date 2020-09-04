#!/usr/bin/env python3

"""
Evaluate.py
This script evaluates a model in the thesis project.

Author      K.Loaiza
Comments    Created: Thursday, May 6, 2020
"""

import os
import sys
import copy
import json
import torch
import myutils
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/224_s181423/categories', help="folder containing the dataset")
parser.add_argument('--model_dir', default='experiments_adam/cm_vgg16_1_adam_lower_lr_one_more_linear_layer', help="folder containing params.json")
parser.add_argument('--net_dir', default='networks_categories', help="folder containing artificial_neural_network.py")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")
parser.add_argument('--file', default='test', help=".csv filename that will be evalutated")
parser.add_argument('--fold', default=1, help="fold number to get mean and std of images, default:1")


def eval(file, dataloaders, dataset_sizes, net):
    """
    Evaluate a net.
    """
    # Reproducibility
    myutils.myseed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load network and restore settings from .tar file
    net = net.to(device)
    fname = f'{args.restore_file}.tar'
    restore_path = os.path.join(args.model_dir, fname)
    checkpoint = torch.load(restore_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()


    # Validation phase
    phase = 'val'
    with torch.no_grad():
        #indexes, predictions, probabilities, all_probabilities, in_labels, in_targets = [],[],[],[],[],[]
        indexes, predictions, probabilities, all_probabilities, in_labels = [],[],[],[],[]
        for index, inputs, labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)

            #_, targets = torch.max(labels, 1)
            probs, preds = torch.max(outputs, 1)

            indexes.extend(index.cpu().detach().numpy())
            all_probabilities.extend(outputs.cpu().detach().numpy())
            probabilities.extend(probs.cpu().detach().numpy())
            predictions.extend(preds.cpu().detach().numpy())
            in_labels.extend(labels.cpu().detach().numpy())
            #in_targets.extend(targets.cpu().detach().numpy())

    #return indexes, probabilities, predictions, all_probabilities, in_labels, in_targets
    return indexes, probabilities, predictions, all_probabilities, in_labels



if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Could not find the dataset at {}".format(args.data_dir)
    assert os.path.isdir(args.model_dir), "Could not find the model at {}".format(args.model_dir)
    assert os.path.isdir(args.net_dir), "Could not find the network at {}".format(args.net_dir)

    # Initialize main log folder
    logs_dir_path = os.path.join(os.getcwd(),'Logs')
    if not os.path.exists(logs_dir_path):
        os.mkdir(logs_dir_path)

    # Initialize main log file
    log_file = os.path.join(logs_dir_path, 'process.log')
    logging_process = myutils.setup_logger(log_file, date=True)

    # Save commandline settings to log
    script_activated = ' '.join(sys.argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging_process.info(f'Script: {script_activated}, device: {device}')

    # Get the mean and std
    with open('dicts/stats_dict.json') as json_file:
        stats_dict = json.load(json_file)

    # Get the experiment parameters
    params_file = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_file), "No json configuration file found at {}".format(params_file)
    params = myutils.Params(params_file)
    params.batch_size = 1

    dfs = {}
    # Load data from .csv file
    fname = os.path.join(args.data_dir, f'{args.file}.csv')
    frame = pd.read_csv(fname)
    dfs['val'] = frame
    # Load data from .csv file
    #tname = os.path.join(args.data_dir, f'{args.tfile}.csv')
    #train = pd.read_csv(tname)
    # Statistics
    fold = args.fold
    mean, std = stats_dict[f'mean{fold}'], stats_dict[f'std{fold}']
    #logging_process.info(f'Model: {args.model_dir}\tFold: {fold}\tTrain: {tname}\tMean: {mean}\tStd: {std}')
    #mean, std = myutils.get_stats(train, params.size)
    #logging_process.info(f'Model: {args.model_dir}\tTrain: {tname}\tMean: {mean}\tStd: {std}')

    # NETWORK SETTINGS
    # Data
    loaders = myutils.get_module(args.net_dir, 'loaders')
    dataloaders, dataset_sizes = loaders.get_loaders(dfs, mean, std, size=params.size, batch_size=params.batch_size, num_workers=params.num_workers)
    # Net
    net = myutils.get_network(args.net_dir, params.network)

    # EVALUATE
    print('-'*10)
    num_steps = len(frame)/params.batch_size
    logging_process.info(f'Model: {args.model_dir}, evaluation has started for {num_steps} steps')
    #indexes, probabilities, predictions, all_probabilities, in_labels, in_targets = eval(args.file, dataloaders, dataset_sizes, net)
    indexes, probabilities, predictions, all_probabilities, in_labels = eval(args.file, dataloaders, dataset_sizes, net)
    logging_process.info(f'Model: {args.model_dir}, evaluation has ended')

    # Save evaluation results to .csv file
    fname = os.path.join(args.data_dir, f'{args.file}.csv')
    df = pd.read_csv(fname)
    df['indexes'] = indexes
    df['probabilities'] = probabilities
    df['predictions'] = predictions
    df['all_probabilities'] = all_probabilities
    df['in_labels'] = in_labels
    #df['targets'] = in_targets
    results_dir = os.path.join(args.model_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fname = fname = os.path.join(results_dir, f'{args.file}.csv')
    df.to_csv(fname)
