"""
Data loading utilities for battery datasets

Supports four major battery degradation datasets:
- XJTU: 55 batteries, 6 charge rate batches
- MIT: 125 batteries, 3 time-based batches
- HUST: 77 batteries, multiple groups
- TJU: 130 batteries, 3 chemistry batches (NCA, NCM, mixed)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from src.utils.logging import write_to_txt


class DF():
    """Base class for data loading with preprocessing capabilities"""

    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method  # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        """Identify outliers using 3-sigma rule"""
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self, df):
        """Remove outliers from dataframe using 3-sigma rule"""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_one_csv(self, file_name, nominal_capacity=None):
        """Read a CSV file and apply preprocessing"""
        df = pd.read_csv(file_name)
        df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0]))

        df = self.delete_3_sigma(df)

        if nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity
            f_df = df.iloc[:, :-1]
            if self.normalization_method == 'min-max':
                f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean()) / f_df.std()

            df.iloc[:, :-1] = f_df

        return df

    def load_one_battery(self, path, nominal_capacity=None):
        """Load a single battery CSV file and create paired consecutive samples"""
        df = self.read_one_csv(path, nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1, y1), (x2, y2)

    def load_all_battery(self, path_list, nominal_capacity):
        """Load multiple battery files and create dataloaders"""
        X1, X2, Y1, Y2 = [], [], [], []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)
            write_to_txt(save_name, 'data path:')
            write_to_txt(save_name, str(path_list))
        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        tensor_X1 = torch.from_numpy(X1).float()
        tensor_X2 = torch.from_numpy(X2).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1, 1)

        # Condition 1: Split train/valid/test (80% train+valid, 20% test)
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]

        # Split train/valid (80/20)
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)

        # Condition 2: Random 80/20 split (for transfer learning)
        train_X1_2, valid_X1_2, train_X2_2, valid_X2_2, train_Y1_2, valid_Y1_2, train_Y2_2, valid_Y2_2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
        train_loader_2 = DataLoader(TensorDataset(train_X1_2, train_X2_2, train_Y1_2, train_Y2_2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)
        valid_loader_2 = DataLoader(TensorDataset(valid_X1_2, valid_X2_2, valid_Y1_2, valid_Y2_2),
                                  batch_size=self.args.batch_size,
                                  shuffle=True)

        # Condition 3: All data as test set (for evaluation)
        test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                 batch_size=self.args.batch_size,
                                 shuffle=False)

        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2, 'valid_2': valid_loader_2,
                  'test_3': test_loader_3}
        return loader


class XJTUdata(DF):
    """XJTU Battery Dataset Loader

    55 batteries across 6 charge rate batches:
    2C, 3C, R2.5, R3, RW, satellite
    Nominal capacity: 2.0 Ah
    """

    def __init__(self, root, args):
        super(XJTUdata, self).__init__(args)
        self.root = root
        self.file_list = os.listdir(root)
        self.variables = pd.read_csv(os.path.join(root, self.file_list[0])).columns
        self.num = len(self.file_list)
        self.batch_names = ['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite']
        self.batch_size = args.batch_size

        if self.normalization:
            self.nominal_capacity = 2.0
        else:
            self.nominal_capacity = None

    def read_one_batch(self, batch='2C'):
        """Load a specific batch (charge rate condition)"""
        if isinstance(batch, int):
            batch = self.batch_names[batch]
        assert batch in self.batch_names, 'batch must be in {}'.format(self.batch_names)
        file_list = []
        for i in range(self.num):
            if batch in self.file_list[i]:
                path = os.path.join(self.root, self.file_list[i])
                file_list.append(path)
        return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)

    def read_all(self, specific_path_list=None):
        """Load all batteries or specific battery files"""
        if specific_path_list is None:
            file_list = []
            for file in self.file_list:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class HUSTdata(DF):
    """HUST Battery Dataset Loader

    77 batteries across multiple groups
    Nominal capacity: 1.1 Ah
    """

    def __init__(self, root, args):
        super(HUSTdata, self).__init__(args)
        self.root = root
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None

    def read_all(self, specific_path_list=None):
        """Load all batteries or specific battery files"""
        if specific_path_list is None:
            file_list = []
            files = os.listdir(self.root)
            for file in files:
                path = os.path.join(self.root, file)
                file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class MITdata(DF):
    """MIT Battery Dataset Loader

    125 batteries across 3 time-based batches
    Nominal capacity: 1.1 Ah
    """

    def __init__(self, root, args):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12', '2017-06-30', '2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None

    def read_one_batch(self, batch):
        """Load a specific time batch (1, 2, or 3)"""
        assert batch in [1, 2, 3], 'batch must be in {}'.format([1, 2, 3])
        root = os.path.join(self.root, self.batchs[batch - 1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root, file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self, specific_path_list=None):
        """Load all batteries or specific battery files"""
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root, batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root, file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)


class TJUdata(DF):
    """TJU Battery Dataset Loader

    130 batteries across 3 chemistry batches:
    Dataset_1_NCA_battery (3.5 Ah)
    Dataset_2_NCM_battery (3.5 Ah)
    Dataset_3_NCM_NCA_battery (2.5 Ah)
    """

    def __init__(self, root, args):
        super(TJUdata, self).__init__(args)
        self.root = root
        self.batchs = ['Dataset_1_NCA_battery', 'Dataset_2_NCM_battery', 'Dataset_3_NCM_NCA_battery']
        if self.normalization:
            self.nominal_capacities = [3.5, 3.5, 2.5]
        else:
            self.nominal_capacities = [None, None, None]

    def read_one_batch(self, batch):
        """Load a specific chemistry batch (1, 2, or 3)"""
        assert batch in [1, 2, 3], 'batch must be in {}'.format([1, 2, 3])
        root = os.path.join(self.root, self.batchs[batch - 1])
        file_list = os.listdir(root)
        path_list = []
        for file in file_list:
            file_name = os.path.join(root, file)
            path_list.append(file_name)
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacities[batch - 1])

    def read_all(self, specific_path_list):
        """Load specific battery files (must specify paths)"""
        for i, batch in enumerate(self.batchs):
            if batch in specific_path_list[0]:
                normal_capacity = self.nominal_capacities[i]
                break
        return self.load_all_battery(path_list=specific_path_list, nominal_capacity=normal_capacity)
