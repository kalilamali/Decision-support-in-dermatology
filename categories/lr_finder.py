#!/usr/bin/env python3

"""
Train.py
This script trains a model in the thesis project.

Author      K.Loaiza
Comments    Created:
"""

# Libraries
import os
import sys
import copy
import json
import torch
import myutils
import argparse
import numpy as np
import pandas as pd

from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/224_s181423/categories', help="folder containing the data, default:/scratch/224_s181423/categories")
parser.add_argument('--model_dir', default='experiments/model1', help="folder containing params.json, default:experiments/model1")
parser.add_argument('--net_dir', default='networks_categories', help="folder containing artificial_neural_network.py, default:networks_categories")
parser.add_argument('--folds', default=3, help='cross validation folds, default:3', type=int)
parser.add_argument('--resume', default=False, help="resume training for more epochs, default:False")
parser.add_argument('--fold_start', default=1, help="change only if resume training from fold different than 1, default:1", type=int)


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
            best_acc = best_checkpoint['acc']

        except FileNotFoundError as err:
            # This error happens when folds are present
            # If interrupted on fold 1 then best best_checkpoint for fold 2 does
            # not exists. This is fixed like this.
            logging_process.info(f'Model: {args.model_dir}\tError: {err}')

    # TRAINING LOOP
    for epoch in range(epoch, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')

        # To track values in each epoch
        tloss,tacc,vloss,vacc = '','','',''

        # Each epoch has a training phase and a validation phase
        for phase in ['train','val']:

            if phase == 'train':
                net.train()  # Set net to training mode

                # Track learning rate for plot
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
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Batch statistics
                running_loss += loss.detach().item() * inputs.size(0)  # This is batch loss
                running_corrects += torch.sum(preds == labels.data)  # This is batch accuracy

            # Epoch statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                tloss = epoch_loss
                tacc = epoch_acc

            if phase == 'val':
                vloss = epoch_loss
                vacc = epoch_acc
                logging_train.info('Epoch: {}\ttloss: {:.4f}\ttacc: {:.4f}\tvloss: {:.4f}\tvacc: {:.4f}'.format(epoch, tloss, tacc, vloss, vacc))

                # Save last settings to .tar file
                torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss
                }, last_path)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc

                    # Save best settings to .tar file
                    torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
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

                #turn off for lr finder
                ##vgg
                #if net_name.startswith('vgg'):
                #    scheduler.step(epoch_acc)

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

    # Get the mean and std
    with open('dicts/stats_dict.json') as json_file:
        stats_dict = json.load(json_file)

    # Get the mean and std
    with open('dicts/weights_dict.json') as json_file:
        weights_dict = json.load(json_file)

    # Get the experiment parameters
    params_file = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(params_file), "No json configuration file found at {}".format(params_file)
    params = myutils.Params(params_file)

    # FOLD LOOP
    dfs = {}
    for fold in range(args.fold_start, args.folds+1):

        # Load data from .csv files
        tname = os.path.join(args.data_dir, f'train{fold}.csv')
        vname = os.path.join(args.data_dir, f'val{fold}.csv')
        train = pd.read_csv(tname)
        print(tname)
        val = pd.read_csv(vname)
        dfs['train'] = train
        dfs['val'] = val

        # Statistics
        mean, std = stats_dict[f'mean{fold}'], stats_dict[f'std{fold}']
        logging_process.info(f'Model: {args.model_dir}\tFold: {fold}\tTrain: {tname}\tMean: {mean}\tStd: {std}')

        # Weights
        weight = weights_dict[f'weights{fold}']
        weight = torch.tensor(weight)
        weight = weight.to(device)
        #weight = None
        # NETWORK SETTINGS

        # Data
        loaders = myutils.get_module(args.net_dir, 'loaders')
        dataloaders, dataset_sizes = loaders.get_loaders(dfs, mean, std, size=params.size, batch_size=params.batch_size, num_workers=params.num_workers)

        # Net
        net = myutils.get_network(args.net_dir, params.network)
        optimizer = myutils.get_optimizer(params.optimizer, net, params.learning_rate, params.momentum, params.weight_decay)

        # Loss function
        criterion = myutils.get_loss_fn(args.net_dir, params.network, weight)
        criterion = criterion.to(device)
        logging_process.info(f'Model: {args.model_dir}\tFile: train{fold}.csv\tWeight: {weight}')

        # FIND LR
        lr_finder = LRFinder(net, optimizer, criterion, device=device)
        trainloader = dataloaders['train']
        valloader = dataloaders['val']
        class CustomTrainIter(TrainDataLoaderIter):
            # My dataloader returns index, X, y
            def inputs_labels_from_batch(self, batch_data):
                return batch_data[1], batch_data[2]

        class CustomValIter(ValDataLoaderIter):
            # My dataloader returns index, X, y
            def inputs_labels_from_batch(self, batch_data):
                return batch_data[1], batch_data[2]
        custom_train_iter = CustomTrainIter(trainloader)
        custom_val_iter = CustomValIter(valloader)
        lr_finder.range_test(custom_train_iter, end_lr=10, num_iter=params.num_epochs, step_mode='exp')
        # Val loader does not work
        #lr_finder.range_test(custom_train_iter, val_loader=custom_val_iter, end_lr=10, num_iter=params.num_epochs, step_mode='exp')
        mylrs = lr_finder.history['lr']
        mylosses = lr_finder.history['loss']
        min_grad_idx = np.gradient(np.array(mylosses)).argmin()
        print(f'Suggested lr: {mylrs[min_grad_idx]}')
        lr_metrics = {'lr': mylrs,'loss': mylosses}
        fname = os.path.join(args.model_dir, f'lr_metrics.json')
        with open (fname, 'w') as f:
            f.write(json.dumps(lr_metrics))
        '''
        # Train
        print(f'Fold {fold}')
        print('-'*10)
        logging_process.info(f'Model: {args.model_dir}\tFold: {fold}, training has started for {params.num_epochs} epochs')
        train_eval(fold, dataloaders, dataset_sizes, net, criterion, optimizer,  scheduler, net_name=params.network, num_epochs=params.num_epochs)
        logging_process.info(f'Model: {args.model_dir}\tFold: {fold}, training has ended')
        '''
