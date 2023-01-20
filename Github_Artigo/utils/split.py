import pandas as pd
from chemprop.data import utils
from random import randint
import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')
seed_val = 42

def get_partition_as_df(partition):
    dict = {'ID': partition.compound_names(), 'SMILES': partition.smiles()}
    df = pd.DataFrame(dict)
    df2 = pd.DataFrame(df)
    #df2[target_names] = pd.DataFrame(df2.targets.values.tolist(), index= df2.index)
    #df2.drop('targets', axis=1, inplace=True)
    return df2


def random_split(dataset, seed_val):
    # get target names (assuming that 1st column contains molecule names and 2nd column contains smiles and rest of the columns are targets)
    df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'ID': str})
    cols = list(df.columns)
    #target_names = cols[2:]

    mol_dataset = utils.get_data(dataset, use_compound_names=True)
    train, valid, test = utils.split_data(mol_dataset, sizes=(0.8, 0.1, 0.1), seed=seed_val)
    train_df = get_partition_as_df(train)
    train_df = train_df[['ID']]
    valid_df = get_partition_as_df(valid)
    valid_df = valid_df[['ID']]
    test_df = get_partition_as_df(test)
    test_df = test_df[['ID']]
    return train_df, valid_df, test_df


def scaffold_split(dataset, seed):
    # get target names (assuming that 1st column contains molecule names and 2nd column contains smiles and rest of the columns are targets)
    df = pd.read_csv(dataset, sep=",", index_col=None, dtype={'ID': str})
    cols = list(df.columns)
    target_names = cols[2:]

    mol_dataset = utils.get_data(dataset, use_compound_names=True)
    train, valid, test = utils.split_data(mol_dataset, split_type="scaffold_balanced", sizes=(0.8, 0.1, 0.1), seed=seed_val)
    train_df = get_partition_as_df(train)
    train_df = train_df[['ID']]
    valid_df = get_partition_as_df(valid)
    valid_df = valid_df[['ID']]
    test_df = get_partition_as_df(test)
    test_df = test_df[['ID']]
    return train_df, valid_df, test_df