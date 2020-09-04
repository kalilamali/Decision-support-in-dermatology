#!/usr/bin/env python3

"""
Train.py
This script trains a model in the thesis project.

Author      K.Loaiza
Comments    Created:  Wednesday, July 15, 2020
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
parser.add_argument('--data_dir', default='data', help="folder containing the data, default:data")
parser.add_argument('--model_dir', default='experiments/model1', help="folder containing params.json, default:experiments/model1")
parser.add_argument('--net_dir', default='networks', help="folder containing artificial_neural_network.py, default:networks")
parser.add_argument('--folds', default=3, help='cross validation folds, default:3', type=int)
parser.add_argument('--resume', default=False, help="resume training for more epochs, default:False")
parser.add_argument('--fold_start', default=1, help="change only if resume training from fold different than 1, default:1", type=int)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.045, lr_decay_epoch=2, factor=0.94):
    """
    Decay learning rate by a factor of 0.94 every lr_decay_epoch epochs.
    """
    lr = init_lr * (factor **(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train_eval(fold, dataloaders, dataset_sizes, net, criterion, optimizer, scheduler, net_name, num_epochs):
    """
    Train and evaluate a net.
    """
    # Initialize logs
    fname = os.path.join(args.model_dir, f'train{fold}.log')
    logging_train = myutils.setup_logger(fname)
    fname = os.path.join(args.model_dir, f'lr{fold}.log')
    logging_lr = myutils.setup_logger(fname)
    # Reproducibility
    myutils.myseed(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load initial weights
    net = net.to(device)
    #best_net_wts = copy.deepcopy(net.state_dict())
    best_acc, epoch = 0.0, 1

    # Initialize .tar files to save settings
    fname = f'last{fold}.tar'
    last_path = os.path.join(args.model_dir, fname)
    fname = f'best{fold}.tar'
    best_path = os.path.join(args.model_dir, fname)

    # To resume training for more epochs
    if args.resume:
        try:
            # Load last settings from .tar file
            last_checkpoint = torch.load(last_path)
            net.load_state_dict(last_checkpoint['net_state_dict'])
            optimizer.load_state_dict(last_checkpoint['optimizer_state_dict'])
            epoch = last_checkpoint['epoch'] + 1  # Since last epoch was saved we start with the next one
            logging_process.info(f'Model: {args.model_dir}\tLast epoch saved: {epoch-1}, resumming training since epoch: {epoch}')

            # Load best settings from .tar file
            best_checkpoint = torch.load(best_path)
            #best_net_wts = best_checkpoint['net_state_dict']
            best_acc = best_checkpoint['acc']

        except FileNotFoundError as err:
            # This error happens when folds are present
            # If interrupted on fold 1 then best best_checkpoint for fold 2 does
            # not exists. This is fixed like this.
            logging_process.info(f'Model: {args.model_dir}\tError: {err}')

    # TRAINING LOOP
    for epoch in range(epoch, num_epochs+1):

        print(f'Epoch {epoch}/{num_epochs}')
        logging_train.info(f'Epoch {epoch}/{num_epochs}')

        # Each epoch has a training phase and a validation phase
        for phase in ['train','val']:
            if phase == 'train':
                net.train()  # Set net to training mode
                mylr_value = optimizer.param_groups[0]['lr']
                logging_lr.info(f'Epoch {epoch}\tlr: {mylr_value}')
            else:
                net.eval()   # Set net to evaluate mode

            # Track statistics
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for index, inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    _, targets = torch.max(labels, 1)
                    _, preds = torch.max(outputs, 1)

                    #if net_name.startswith('vgg16_ft_no_soft'):
                    #    outputs = torch.reshape(outputs, (-1,)) # reshape added for binary
                    #    loss = criterion(outputs, targets.float()) # float added for binary

                    #else:
                    loss = criterion(outputs, targets)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Batch statistics
                running_loss += loss.detach().item() * inputs.size(0)  # This is batch loss
                running_corrects += torch.sum(preds == targets.data)  # This is batch accuracy

            # efficientnetb
            if net_name.startswith('efficientnetb'):
                if phase == 'train':
                    scheduler.step()

            # inceptionv
            if net_name.startswith('inceptionv'):
                if phase == 'train':
                    if (epoch % 2) == 0:
                        scheduler.step()

            # Epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging_train.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':

                # Save last settings to .tar file
                torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, last_path)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_net_wts = net.state_dict()


                    # Save best settings to .tar file
                    torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(), #best_net_wts
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'acc': best_acc
                    }, best_path)

                    # Save best settings to .json file
                    best_metrics = {
                    f'loss{fold}': epoch_loss,
                    f'acc{fold}': best_acc.item()
                    }
                    fname = os.path.join(args.model_dir, f'metrics{fold}.json')
                    with open (fname, 'w') as f:
                        f.write(json.dumps(best_metrics))

                # vgg
                if net_name.startswith('vgg'):
                    scheduler.step(epoch_acc)

                # resnet
                if net_name.startswith('resnet'):
                    scheduler.step(epoch_loss)

    print('Best val Acc: {:4f}'.format(best_acc))
    logging_process.info('Model: {}\tFold: {}\tBest val Acc: {:4f}'.format(args.model_dir, fold, best_acc))


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

    # Get the experiment parameters
    params_file = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_file), "No json configuration file found at {}".format(params_file)
    params = myutils.Params(params_file)

    # FOLD LOOP
    dfs = {}
    for fold in range(args.fold_start, args.folds+1):
        # Load data from .csv files
        tname = os.path.join(args.data_dir, 'train.csv')
        vname = os.path.join(args.data_dir, 'val.csv')

        if args.folds > 1:
            # Load data from .csv files
            tname = os.path.join(args.data_dir, f'train{fold}.csv')
            vname = os.path.join(args.data_dir, f'val{fold}.csv')

        train = pd.read_csv(tname)
        print(tname)
        val = pd.read_csv(vname)
        dfs['train'] = train
        dfs['val'] = val
        mean, std = myutils.get_stats(train, params.size)
        logging_process.info(f'Model: {args.model_dir}\tFold: {fold}\tTrain: {tname}\tMean: {mean}\tStd: {std}')

        # NETWORK SETTINGS
        # Data
        loaders = myutils.get_module(args.net_dir, 'loaders')
        dataloaders, dataset_sizes = loaders.get_loaders(dfs, mean, std, size=params.size, batch_size=params.batch_size, num_workers=params.num_workers)
        # Net
        net = myutils.get_network(args.net_dir, params.network)
        optimizer = myutils.get_optimizer(params.optimizer, net, params.learning_rate, params.momentum, params.weight_decay)
        # Schedulers
        #vgg
        if params.network.startswith('vgg'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max')
        #efficientnetb
        if params.network.startswith('efficientnetb'):
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2.4, gamma=0.97)
        #resnet
        if params.network.startswith('resnet'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min')
        #inceptionv
        if params.network.startswith('inception'):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
        # Loss function
        weight = myutils.get_weight(train)
        weight = weight.to(device)
        #if params.network == 'vgg16_ft_no_soft':
        #    weight = None
        criterion = myutils.get_loss_fn(args.net_dir, params.network, weight)
        criterion = criterion.to(device)
        logging_process.info(f'Model: {args.model_dir}\tFile: train{fold}.csv\tWeight: {weight}')

        # Train
        print(f'Fold {fold}')
        print('-'*10)
        logging_process.info(f'Model: {args.model_dir}\tFold: {fold}, training has started for {params.num_epochs} epochs')
        train_eval(fold, dataloaders, dataset_sizes, net, criterion, optimizer,  scheduler, net_name=params.network, num_epochs=params.num_epochs)
        logging_process.info(f'Model: {args.model_dir}\tFold: {fold}, training has ended')
