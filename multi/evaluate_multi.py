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
parser.add_argument('--data_dir', default='data/binary', help="folder containing the dataset")
parser.add_argument('--model_dir', default='experiments/model1', help="folder containing params.json")
parser.add_argument('--net_dir', default='networks_binary', help="folder containing artificial_neural_network.py")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir containing weights to load")
#parser.add_argument('--tfile', default='train', help=".csv filename to calculate mean and std")
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

        bins_predictions, bins_probabilities, bins_all_probabilities, bins_in_labels = [],[],[],[]
        cats_predictions, cats_probabilities, cats_all_probabilities, cats_in_labels = [],[],[],[]
        indexes = []

        for index, inputs, bins_labels, cats_labels in tqdm(dataloaders[phase]):
            inputs = inputs.to(device)
            bins_labels = bins_labels.to(device)
            cats_labels = cats_labels.to(device)

            outputs = net(inputs)
            bins_outputs = outputs[0]
            cats_outputs = outputs[1]

            #_, targets = torch.max(labels, 1)
            #_, bins_targets = torch.max(bins_labels, 1)
            #_, cats_targets = torch.max(cats_labels, 1)
            #probs, preds = torch.max(outputs, 1)
            bins_probs, bins_preds = torch.max(bins_outputs, 1)
            cats_probs, cats_preds = torch.max(cats_outputs, 1)

            indexes.extend(index.cpu().detach().numpy())

            bins_all_probabilities.extend(bins_outputs.cpu().detach().numpy())
            bins_probabilities.extend(bins_probs.cpu().detach().numpy())
            bins_predictions.extend(bins_preds.cpu().detach().numpy())
            bins_in_labels.extend(bins_labels.cpu().detach().numpy())
            #bins_in_targets.extend(bins_targets.cpu().detach().numpy())

            cats_all_probabilities.extend(cats_outputs.cpu().detach().numpy())
            cats_probabilities.extend(cats_probs.cpu().detach().numpy())
            cats_predictions.extend(cats_preds.cpu().detach().numpy())
            cats_in_labels.extend(cats_labels.cpu().detach().numpy())
            #cats_in_targets.extend(cats_targets.cpu().detach().numpy())

    B = [bins_probabilities, bins_predictions, bins_all_probabilities, bins_in_labels]
    C = [cats_probabilities, cats_predictions, cats_all_probabilities, cats_in_labels]
    return indexes, B,C


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
    indexes, B, C = eval(args.file, dataloaders, dataset_sizes, net)
    bins_probabilities, bins_predictions, bins_all_probabilities, bins_in_labels = B
    cats_probabilities, cats_predictions, cats_all_probabilities, cats_in_labels = C
    logging_process.info(f'Model: {args.model_dir}, evaluation has ended')

    # Save evaluation results to .csv file
    fname = os.path.join(args.data_dir, f'{args.file}.csv')
    df = pd.read_csv(fname)
    df['indexes'] = indexes
    df['bins_probabilities'] = bins_probabilities
    df['bins_predictions'] = bins_predictions
    df['bins_all_probabilities'] = bins_all_probabilities
    df['bins_in_labels'] = bins_in_labels


    df['cats_probabilities'] = cats_probabilities
    df['cats_predictions'] = cats_predictions
    df['cats_all_probabilities'] = cats_all_probabilities
    df['cats_in_labels'] = cats_in_labels


    results_dir = os.path.join(args.model_dir, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    fname = fname = os.path.join(results_dir, f'{args.file}.csv')
    df.to_csv(fname)
