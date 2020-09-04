#!/usr/bin/env python3

"""
Build_data.py
This script builds the data in the thesis project.

Author      K.Loaiza
Comments    Created: Wednesday, July 15, 2020
"""

import os
import json
import sys
import torch
import argparse
import myutils
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="folder containing the data, default:data")
parser.add_argument('--folds', default=3, help='cross validation folds, default:3', type=int)


def load_data(data_dir):
    """
    Function that takes a folder, finds all .jpg files inside the folder,
    and creates a dataframe.
    """
    # Reproducibility
    myutils.myseed(seed=42)

    # Get the image paths
    filenames = myutils.run_fast_scandir(data_dir, [".jpg"])
    df = pd.DataFrame(data=filenames, columns=['filenames'])

    # Get the label from nth folder starting from the parent:
    outlevel = 4  # fname = '/scratch/s181423_data/data_bin/label/image.jpg'
    df['label'] = df['filenames'].apply(lambda x: x.split('/')[outlevel])

    # Get the id from the basename
    df['id'] = df['filenames'].apply(lambda x: os.path.basename(x))

    # Get label as one hot encoded values
    df = df.set_index(['id','filenames'])
    df['label'] = df['label'].astype('category')
    df = pd.get_dummies(df, prefix='', prefix_sep='')
    df = df.reset_index()

    # Save the data as a .csv file
    df.to_csv(f'{data_dir}.csv', index=False)
    logging_data_process.info(f'Saved: {data_dir}.csv')


def data_split(data_dir, folds):
    """
    Function that takes a data_dir and a number of folds,
    and splits images in data_dir into
    training(80%) and testing(20%) data.

    For fit.py training data is further splitted into
    training and validation sets.

    If cross validation is needed, training data is also splitted into
    train and validation folds.
    """
    # Reproducibility
    myutils.myseed(seed=42)
    seed = 42

    # Load the data with image paths and labels
    df = pd.read_csv(f'{data_dir}.csv')
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logging_data_process.info(f"all data size:{len(df)}")

    # Test
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    train_val, test = train_val.reset_index(drop=True), test.reset_index(drop=True)
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    logging_data_process.info(f"test size:{len(test)}")
    logging_data_process.info(f"train_val size:{len(train_val)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'test.csv')}")

    # Train and validation
    #X = train_val[['id','filenames']]
    #y = train_val.iloc[:,2:].apply(lambda x: np.argmax(x), axis=1)  # argmax is necessary for stratification
    # categories did not allow stratify because some classes have just 1 example
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed, shuffle=True)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
    logging_data_process.info(f"train size:{len(train)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'train.csv')}")
    logging_data_process.info(f"val size:{len(val)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'val.csv')}")

    # Cross validation folds
    if folds > 1:
        logging_data_process.info(f'Folds: {folds}')
        X = train_val[['id','filenames']]
        y = train_val.iloc[:,2:].apply(lambda x: np.argmax(x), axis=1)  # argmax is necessary for stratification
        skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        fold = 0
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            train_idx, val_idx = list(train_idx), list(val_idx)
            train, val = train_val.iloc[train_idx,:], train_val.iloc[val_idx,:]
            train, val = train.reset_index(drop=True), val.reset_index(drop=True)
            train.to_csv(os.path.join(data_dir, f'train{fold}.csv'), index=False)
            val.to_csv(os.path.join(data_dir, f'val{fold}.csv'), index=False)
            logging_data_process.info(f"train{fold} size:{len(train)}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'train{fold}.csv')}")
            logging_data_process.info(f"val{fold} size:{len(val)}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'val{fold}.csv')}")


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Could not find the dataset at {}".format(args.data_dir)

    # Initialize main log folder
    logs_dir_path = os.path.join(os.getcwd(),'Logs')
    if not os.path.exists(logs_dir_path):
        os.mkdir(logs_dir_path)

    # Initialize main log file
    log_file = os.path.join(logs_dir_path, 'data.log')
    logging_data_process = myutils.setup_logger(log_file, date=True)

    # Save commandline settings to log
    script_activated = ' '.join(sys.argv)
    logging_data_process.info(f'Script: {script_activated}')

    # Build dataset
    logging_data_process.info('Script: load_data')
    load_data(args.data_dir)
    logging_data_process.info('Script: data_split')
    data_split(args.data_dir, args.folds)

    # DONE
    print(f'Done building dataset, saved to {args.data_dir}')
