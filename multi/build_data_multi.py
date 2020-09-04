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
import sklearn


from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/224_s181423/multi', help="folder containing the data, default:/scratch/224_s181423/multi")
parser.add_argument('--folds', default=1, help='cross validation folds, default:1', type=int)
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

    #   BINARY
    dfa = pd.DataFrame(data=filenames, columns=['filenames'])
    # Get the label from nth folder starting from the parent:
    outlevel = 4  # fname = '/scratch/s181423_data/data_bin/label/image.jpg'
    dfa['label0'] = dfa['filenames'].apply(lambda x: x.split('/')[outlevel])
    # Get the id from the basename
    dfa['id'] = dfa['filenames'].apply(lambda x: os.path.basename(x))
    # Get label as one hot encoded values
    dfa = dfa.set_index(['id','filenames'])
    #dfa['label0'] = dfa['label0'].astype('category')

    # Create a subset of skin to get the disease categories only for these pictures
    df_skin_only = dfa[dfa['label0']=='skin']

    #   CATEGORIES
    dfb = pd.DataFrame(data=filenames, columns=['filenames'])
    # Get the label from nth folder starting from the parent:
    outlevel = 5  # fname = '/scratch/s181423_data/data_bin/label/image.jpg'
    dfb['label1'] = dfb['filenames'].apply(lambda x: x.split('/')[outlevel])
    # Get the id from the basename
    dfb['id'] = dfb['filenames'].apply(lambda x: os.path.basename(x))
    # Get label as one hot encoded values
    dfb = dfb.set_index(['id','filenames'])
    #dfb['label1'] = dfb['label1'].astype('category')

    # Get disease categories only for the skin images
    df_diseases = pd.concat([df_skin_only, dfb], axis=1, sort=False, join='inner').drop(['label0'], axis=1)

    # Join binary and categories labels
    df = pd.concat([dfa, df_diseases], axis=1, sort=False)
    df = df.fillna('AAA')
    df['label0'] = df['label0'].astype('category')
    df['label1'] = df['label1'].astype('category')
    mapping = {}
    mapping_binary = dict(enumerate(df['label0'].cat.categories ))
    mapping_categories = dict(enumerate(df['label1'].cat.categories ))
    df['label0'] = pd.Categorical(df['label0']).codes
    df['label1'] = pd.Categorical(df['label1']).codes

    mapping['mapping_binary'] = mapping_binary
    mapping['mapping_categories'] = mapping_categories
    #df = pd.get_dummies(df, prefix='', prefix_sep='')
    df = df.reset_index()

    # Resample the minority classes to have them in training and testing
    counts_df = pd.DataFrame(df['label1'].value_counts())
    labels_with_one_example = list(counts_df[counts_df['label1'] < 2].index)
    duplicates_df = df[df['label1'].isin(labels_with_one_example)]
    # 5-plicate df for stratify 20% of 5 is 1, for 80% train, 20% test
    df_copy = duplicates_df
    df = pd.concat([df, df_copy, df_copy, df_copy, df_copy])

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

    For fit.py training data is further splitted into
    training and validation sets.

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
    #df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logging_data_process.info(f"all data size:{len(df)}")

    # Test
    y = list(df['label1'])
    train_val, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    train_val, test = train_val.reset_index(drop=True), test.reset_index(drop=True)

    # Remove samples used in training from test.csv
    ids = list(train_val.id)
    test = test[~test.id.isin(ids)]
    test

    print(f'test:{len(test.label1.value_counts())}')
    test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
    logging_data_process.info(f"test size:{len(test)}")
    logging_data_process.info(f"train_val size:{len(train_val)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, 'test.csv')}")

    if folds == 1:
        logging_data_process.info(f'Folds: {folds}')
        fold = 1
    # Train and validation
    y = list(train_val['label1'])
    #X = train_val[['id','filenames']]
    #y = train_val.iloc[:,2:].apply(lambda x: np.argmax(x), axis=1)  # argmax is necessary for stratification
    # categories did not allow stratify because some classes have just 1 example
    train, val = train_test_split(train_val, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)

    size = len(train['label1'].unique()) * 1000
    print(f'size: {size}')
    train = rep_sample(train, 'label1', size)
    train = sklearn.utils.shuffle(train)
    print(f'train{fold}:{len(train.label1.value_counts())}')
    print(f'val{fold}:{len(val.label1.value_counts())}')

    train.to_csv(os.path.join(data_dir, f'train{fold}.csv'), index=False)
    val.to_csv(os.path.join(data_dir, f'val{fold}.csv'), index=False)
    logging_data_process.info(f"train{fold} size:{len(train)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, f'train{fold}.csv')}")
    logging_data_process.info(f"val{fold} size:{len(val)}")
    logging_data_process.info(f"Saved: {os.path.join(data_dir, f'val{fold}.csv')}")


    # Cross validation folds
    if folds > 1:
        print('WARNING: Karen has not implemented this yet!')
    '''
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
    '''


def get_all_stats(data_dir, folds, size=224):
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

    # DONE
    print(f'Done building dataset, saved to {args.data_dir}')
