import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np


class ABDataset(Dataset):
    def __init__(self, chains, labels, cdrs):
        self.chains = chains
        self.labels = labels
        self.cdrs = cdrs

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        return self.chains[idx], self.labels[idx], self.cdrs[idx]


def ab_loader(train, valid, test):
    return DataLoader(train, batch_size=8), DataLoader(valid, batch_size=8), DataLoader(test, batch_size=8)


def get_dataloaders(test_df, train_df, val_df, max_len):
    def to_binary(input_list):
        return np.array([0. if c == '0' else 1. for c in input_list] + [0. for _ in range(max_len - len(input_list))])

    def to_fixed_len(input_list):
        input_list = input_list.strip('][').split(', ')
        int_input_list = [int(i) for i in input_list]
        return np.array([1 if (i in int_input_list) else 0 for i in range(max_len)])

    chains_train = [x for x in train_df['sequence'].tolist()]
    labels_train = [to_binary(x) for x in train_df['paratope'].tolist()]
    cdrs_train = [to_fixed_len(idx) for idx in train_df['cdrs'].tolist()]
    chains_valid = [x for x in val_df['sequence'].tolist()]
    labels_valid = [to_binary(x) for x in val_df['paratope'].tolist()]
    cdrs_valid = [to_fixed_len(idx) for idx in val_df['cdrs'].tolist()]
    chains_test = [x for x in test_df['sequence'].tolist()]
    labels_test = [to_binary(x) for x in test_df['paratope'].tolist()]
    cdrs_test = [to_fixed_len(idx) for idx in test_df['cdrs'].tolist()]
    train_data = ABDataset(chains_train, labels_train, cdrs_train)
    valid_data = ABDataset(chains_valid, labels_valid, cdrs_valid)
    test_data = ABDataset(chains_test, labels_test, cdrs_test)
    return ab_loader(train_data, valid_data, test_data)


def get_single_dataloaders(max_len):
    df = pd.read_csv("data/processed_dataset.csv")

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    return get_dataloaders(test_df, train_df, val_df, max_len)


def get_cv_dataloaders(cross_round, max_len):
    df = pd.read_csv("data/processed_dataset.csv")

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    train_index, test_index = list(kf.split(range(len(df))))[cross_round]
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    return get_dataloaders(test_df, train_df, val_df, max_len)
