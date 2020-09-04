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
parser.add_argument('--data_dir', default='/scratch/224_s181423/multi', help="folder containing the data, default:/scratch/224_s181423/multi")
parser.add_argument('--model_dir', default='experiments_multi', help="folder containing params.json, default:experiments_multi")
parser.add_argument('--net_dir', default='networks_multi', help="folder containing artificial_neural_network.py, default:networks_multi")
parser.add_argument('--folds', default=1, help='cross validation folds, default:1', type=int)
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
    fname = os.path.join(args.model_dir, f'bins{fold}.log')
    logging_bins = myutils.setup_logger(fname)
    fname = os.path.join(args.model_dir, f'cats{fold}.log')
    logging_cats = myutils.setup_logger(fname)

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

        # To track values in each epoch
        tloss,tacc,vloss,vacc = '','','',''
        tloss0,tacc0,vloss0,vacc0 = '','','',''
        tloss1,tacc2,vloss3,vacc4 = '','','',''


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
            running_loss0 = 0.0
            running_loss1 = 0.0

            running_corrects0 = 0
            running_corrects1= 0

            # Iterate over data
            for index, inputs, bins_labels, cats_labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                bins_labels = bins_labels.to(device)
                cats_labels = cats_labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs_bins, outputs_cats = net(inputs)
                    #outputs_bins = torch.reshape(outputs_bins, (-1,)) # reshape added for binary
                    outputs_bins = outputs_bins.to(device)
                    outputs_cats = outputs_cats.to(device)

                    #loss0 = criterion[0](outputs_bins, bins_labels.float())# float added for binary
                    loss0 = criterion[0](outputs_bins, bins_labels)
                    loss1 = criterion[1](outputs_cats, cats_labels)
                    loss0 = loss0 * (2/307)
                    loss1 = loss1 * (305/307)

                    #loss0 = loss0 * (2/306)
                    #loss1 = loss1 * (304/306)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss = (loss0 + loss1)/2
                        loss.backward()
                        optimizer.step()

                # Batch statistics
                running_loss0 += loss0.detach().item() * inputs.size(0)
                running_loss1 += loss1.detach().item() * inputs.size(0)

                #running_corrects0 += torch.sum(torch.round(outputs_bins) == bins_labels.data)
                running_corrects0 += torch.sum(torch.max(outputs_bins,1)[1] == bins_labels.data)
                running_corrects1 += torch.sum(torch.max(outputs_cats, 1)[1] == cats_labels.data)

            # efficientnetb
            #if net_name.startswith('efficientnetb'):
            #    if phase == 'train':
            #        scheduler.step()

            # inceptionv
            #if net_name.startswith('inceptionv'):
            #    if phase == 'train':
            #        if (epoch % 2) == 0:
            #            scheduler.step()

            # Epoch statistics
            epoch_loss0 = running_loss0 / dataset_sizes[phase]
            epoch_loss1 = running_loss1 / dataset_sizes[phase]

            epoch_loss = epoch_loss0 + epoch_loss1

            epoch_acc0 = (running_corrects0.double() / dataset_sizes[phase]) * (2/307)
            epoch_acc1 = (running_corrects1.double() / dataset_sizes[phase]) * (305/307)

            epoch_acc = (epoch_acc0 + epoch_acc1)/2

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            #logging_train.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            print('{} bin_loss: {:.4f} bin_acc: {:.4f}'.format(phase, epoch_loss0, epoch_acc0))
            #logging_bins.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss0, epoch_acc0))

            print('{} cat_loss: {:.4f} cat_acc: {:.4f}'.format(phase, epoch_loss1, epoch_acc1))
            #logging_cats.info('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss1, epoch_acc1))
            if phase == 'train':
                tloss = epoch_loss
                tloss0 = epoch_loss0
                tloss1 = epoch_loss1

                tacc = epoch_acc
                tacc0 = epoch_acc0
                tacc1 = epoch_acc1


            if phase == 'val':
                vloss = epoch_loss
                vloss0 = epoch_loss0
                vloss1 = epoch_loss1

                vacc = epoch_acc
                vacc0 = epoch_acc0
                vacc1 = epoch_acc1

                logging_train.info('Epoch: {}\ttloss: {:.4f}\ttacc: {:.4f}\tvloss: {:.4f}\tvacc: {:.4f}'.format(epoch, tloss, tacc, vloss, vacc))
                logging_bins.info('Epoch: {}\ttloss: {:.4f}\ttacc: {:.4f}\tvloss: {:.4f}\tvacc: {:.4f}'.format(epoch, tloss0, tacc0, vloss0, vacc0))
                logging_cats.info('Epoch: {}\ttloss: {:.4f}\ttacc: {:.4f}\tvloss: {:.4f}\tvacc: {:.4f}'.format(epoch, tloss1, tacc1, vloss1, vacc1))


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

                #vgg
                if net_name.startswith('vgg'):
                    scheduler.step(epoch_acc)

                # resnet
                #if net_name.startswith('resnet'):
                #    scheduler.step(epoch_loss)

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
        #weights_list = myutils.get_weight_multi(train)
        #weights_list = [torch.tensor(a), torch.tensor(b)]
        #weights_list = [weights_list[0].to(device), weights_list[1].to(device)]
        weights_list = [None, None]

        # NETWORK SETTINGS
        # Data
        loaders = myutils.get_module(args.net_dir, 'loaders')
        dataloaders, dataset_sizes = loaders.get_loaders(dfs, mean, std, size=params.size, batch_size=params.batch_size, num_workers=params.num_workers)
        # Net
        net = myutils.get_network(args.net_dir, params.network)
        optimizer = myutils.get_optimizer(params.optimizer, net, params.learning_rate, params.momentum, params.weight_decay)
        ## Schedulers
        ##vgg
        #if params.network.startswith('vgg'):
        #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='max')
        #efficientnetb
        #if params.network.startswith('efficientnetb'):
        #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2.4, gamma=0.97)
        #resnet
        #if params.network.startswith('resnet'):
        #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min')
        #inceptionv
        #if params.network.startswith('inception'):
        #    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)

        # Loss function
        criterion = myutils.get_loss_fn(args.net_dir, params.network, weights_list)
        logging_process.info(f'Model: {args.model_dir}\tFile: train{fold}.csv\tWeight: {weights_list}\tbinary_weight: {weights_list[0]}\tcategories_weight: {weights_list[1]}')

        # MY LR FINDER
        mylrs, mylosses = [],[]
        for epoch in range(1):
            lr_lambda = lambda epoch: 5
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda, last_epoch=-1)
            for batch_idx, (index, data, targets0, targets1) in enumerate(dataloaders['train']):
                print('Epoch {}, Batch idx {}, lr {}'.format(epoch, batch_idx, optimizer.param_groups[0]['lr']))
                mylrs.append(optimizer.param_groups[0]['lr'])
                optimizer.zero_grad()
                outputs0, outputs1 = net(data)
                loss0 = criterion[0](outputs0, targets0)
                loss1 = criterion[1](outputs1, targets1)
                loss0 = loss0 * (2/307)
                loss1 = loss1 * (305/307)
                loss = (loss0 + loss1)/2
                mylosses.append(loss0.detach().item())
                loss.backward()
                optimizer.step()
                scheduler.step()
                if optimizer.param_groups[0]['lr'] > 1:
                    break
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
