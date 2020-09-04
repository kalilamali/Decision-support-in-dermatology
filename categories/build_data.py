#!/usr/bin/env python3

"""
Build_data.py
This script builds the data.

Author      K.Loaiza
Comments    Created:
"""

# Libraries
import os
import sys
import json
import torch
import argparse
import myutils
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/224_s181423/categories', help="folder containing the data, default:/scratch/224_s181423/categories")
parser.add_argument('--folds', default=3, help='cross validation folds, default:3', type=int)
parser.add_argument('--size', default=224, help='size of the pictures, default:224', type=int)


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
    outlevel = 4  # fname = '/scratch/s181423/data/label/image.jpg'
    df['label'] = df['filenames'].apply(lambda x: x.split('/')[outlevel])

    # Resample the minority classes to have them in training and testing
    counts_df = pd.DataFrame(df['label'].value_counts())
    labels_with_one_example = list(counts_df[counts_df['label'] < 2].index)
    duplicates_df = df[df['label'].isin(labels_with_one_example)]
    # 5-plicate df for stratify 20% of 5 is 1, for 80% train, 20% test
    df_copy = duplicates_df
    df = pd.concat([df, df_copy, df_copy, df_copy, df_copy])

    # Get the id from the basename
    df['id'] = df['filenames'].apply(lambda x: os.path.basename(x))

    # Get label as one hot encoded values
    df = df.set_index(['id','filenames'])
    df['label'] = df['label'].astype('category')
    mapping = dict(enumerate(df['label'].cat.categories ))
    df['label'] = pd.Categorical(df['label']).codes
    #df = pd.get_dummies(df, prefix='', prefix_sep='')
    df = df.reset_index()

    # Save the data as a .csv file
    df.to_csv(f'{data_dir}.csv', index=False)
    logging_data_process.info(f'Saved: {data_dir}.csv')

    # Save the mappings as a .json file
    with open('dicts/mapping.json', 'w') as f:
        f.write(json.dumps(mapping))
        logging_data_process.info('Saved: dicts/mapping.json')


def data_split(data_dir, folds):
    """
    Function that takes a data_dir and a number of folds,
    and splits images in data_dir into
    training(80%) and testing(20%) data.

    If cross validation is needed, training data is also splitted into
    train and validation folds.
    """
    def rep_sample(df, col, n, *args, **kwargs):
        nu = df[col].nunique()
        m = len(df)
        mpb = n // nu
        mku = n - mpb * nu
        fills = np.zeros(nu)
        fills[:mku] = 1
        sample_sizes = (np.ones(nu) * mpb + fills).astype(int)
        gb = df.groupby(col)
        sample = lambda sub_df, i: sub_df.sample(sample_sizes[i], *args, **kwargs, replace=True)
        subs = [sample(sub_df, i) for i, (_, sub_df) in enumerate(gb)]
        return pd.concat(subs)

    # Reproducibility
    myutils.myseed(seed=42)
    seed = 42

    # Load the data with image paths and labels
    df = pd.read_csv(f'{data_dir}.csv')
    logging_data_process.info(f"all data size:{len(df)}")

    # Test
    y = list(df['label'])
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    train_val, test = train_val.reset_index(drop=True), test.reset_index(drop=True)

    # Remove samples used in training from test.csv
    ids = list(train_val.id)
    test = test[~test.id.isin(ids)]
    test

    print(f'test:{len(test.label.value_counts())}')
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    logging_data_process.info(f"test size:{len(test)}")
    logging_data_process.info(f"train_val size:{len(train_val)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'test.csv')}")

    # Cross validation folds
    if folds == 1:
        logging_data_process.info(f'Folds: {folds}')
        X = train_val[['id','filenames']]
        y = list(train_val['label'])
        n_splits = 2 # Put 2 and break when fold 1 finishes
        #skf = StratifiedKFold(n_splits, random_state=seed, shuffle=True)
        skf = StratifiedShuffleSplit(n_splits, random_state=seed, test_size=0.2)
        fold = 0
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            train_idx, val_idx = list(train_idx), list(val_idx)
            train, val = train_val.iloc[train_idx,:], train_val.iloc[val_idx,:]
            train, val = train.reset_index(drop=True), val.reset_index(drop=True)

            # To overfit to the first balanced batch
            #size = 304
            #train = rep_sample(train, 'label', size)
            #train =sklearn.utils.shuffle(train)
            print(f'train{fold}:{len(train.label.value_counts())}')
            train.to_csv(os.path.join(data_dir, f'train{fold}.csv'), index=False)
            val.to_csv(os.path.join(data_dir, f'val{fold}.csv'), index=False)
            logging_data_process.info(f"train{fold} size:{len(train)}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'train{fold}.csv')}")
            logging_data_process.info(f"val{fold} size:{len(val)}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'val{fold}.csv')}")
            break

    if folds > 1:
        logging_data_process.info(f'Folds: {folds}')
        X = train_val[['id','filenames']]
        y = list(train_val['label'])
        #skf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        skf = StratifiedShuffleSplit(n_splits=folds, random_state=seed, test_size=0.2)
        fold = 0
        for train_idx, val_idx in skf.split(X, y):
            fold += 1
            train_idx, val_idx = list(train_idx), list(val_idx)
            train, val = train_val.iloc[train_idx,:], train_val.iloc[val_idx,:]
            train, val = train.reset_index(drop=True), val.reset_index(drop=True)

            # To overfit to the first balanced batch
            #size = 304
            #train = rep_sample(train, 'label', size)
            #train =sklearn.utils.shuffle(train)
            print(f'train{fold}:{len(train.label.value_counts())}')
            train.to_csv(os.path.join(data_dir, f'train{fold}.csv'), index=False)
            val.to_csv(os.path.join(data_dir, f'val{fold}.csv'), index=False)
            logging_data_process.info(f"train{fold} size:{len(train)}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'train{fold}.csv')}")
            logging_data_process.info(f"val{fold} size:{len(val)}")
            logging_data_process.info(f"Saved: {os.path.join(data_dir, f'val{fold}.csv')}")


def get_all_stats(data_dir, folds, size):
    """
    Function that creates a dictionary with the RGB means and stds.
    """
    # Reproducibility
    myutils.myseed(seed=42)
    stats_dict = {}
    for fold in range(1, folds+1):
        path = os.path.join(data_dir, f'train{fold}.csv')
        train = pd.read_csv(path)
        mean, std = myutils.get_stats(train, size)
        stats_dict[f'mean{fold}'] = mean
        stats_dict[f'std{fold}'] = std

    # Save stats_dict to a .json file
    with open('dicts/stats_dict.json', 'w') as f:
        f.write(json.dumps(stats_dict))
        logging_data_process.info('Saved: dicts/stats_dict.json')


def get_all_weights(data_dir, folds):
    """
    Function that creates a dictionary with the RGB means and stds.
    """
    # Reproducibility
    myutils.myseed(seed=42)
    weights_dict = {}
    for fold in range(1, folds+1):
        path = os.path.join(data_dir, f'train{fold}.csv')
        train = pd.read_csv(path)
        weights = myutils.get_weights(train)
        weights_dict[f'weights{fold}'] = weights

    # Save stats_dict to a .json file
    with open('dicts/weights_dict.json', 'w') as f:
        f.write(json.dumps(weights_dict))
        logging_data_process.info('Saved: dicts/weights_dict.json')


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Could not find the dataset at {}".format(args.data_dir)

    # Initialize folder for .json files
    path = os.path.join(os.getcwd(),'dicts')
    if not os.path.exists(path):
        os.mkdir(path)

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

    logging_data_process.info('Script: get_all_stats')
    get_all_stats(args.data_dir, args.folds, args.size)

    logging_data_process.info('Script: get_all_weights')
    get_all_weights(args.data_dir, args.folds)

    # DONE
    print(f'Done building dataset, saved to {args.data_dir}')
