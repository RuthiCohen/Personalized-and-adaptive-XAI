import os
import torch
import pandas as pd
from errno import EEXIST
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

target_label = {
    'heart_failure_clinical_records_dataset': 'DEATH_EVENT',
}
class TabularDataLoader(data.Dataset):
    #todo: fixed class!!
    def __init__(self, path, filename, label, scale='minmax'):
        self.dataset = pd.read_csv(path + filename)
        self.target = label

        self.X = self.dataset.drop(self.target, axis=1)
        self.feature_names = self.X.columns.to_list()
        self.target_name = label

        # Transform data
        if scale == 'minmax':
            self.scaler = MinMaxScaler()
        elif scale == 'standard':
            self.scaler = StandardScaler()
        elif scale == 'none':
            self.scaler = None
        else:
            raise NotImplementedError(
                'The current version of DataLoader class only provides the following transformations: {minmax, standard, none}')
        if self.scaler is not None:
            self.scaler.fit_transform(self.X)
            self.data = self.scaler.transform(self.X)
        else:
            self.data = self.X.values

        self.targets = self.dataset[self.target]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        idx = idx.tolist() if isinstance(idx, torch.Tensor) else idx
        return (self.data[idx], self.targets.values[idx])

    def get_number_of_features(self):
        return self.data.shape[1]

    def get_number_of_instances(self):
        return self.data.shape[0]

    def mkdir_p(self, mypath):
        """Creates a directory. equivalent to using mkdir -p on the command line"""
        try:
            os.makedirs(mypath)
        except OSError as exc:  # Python >2.5
            if exc.errno == EEXIST and os.path.isdir(mypath):
                pass
            else:
                raise


def ReturnLoaders(data_name, batch_size=32, scaler='minmax'):
    #todo: my code
    prefix = f"data/{data_name}/"
    file_train, file_test = 'train.csv', 'test.csv'

    label = target_label[data_name]
    dataset_train = TabularDataLoader(path=prefix,filename=file_train, label=label, scale=scaler)
    dataset_test = TabularDataLoader(path=prefix,filename=file_test, label=label, scale=scaler)

    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def ReturnTrainTestX(data_name, n_test=None, n_train=None, float_tensor=False):
    #todo: my code
    trainloader, testloader = ReturnLoaders(data_name)
    X_test = testloader.dataset.data[:n_test] if n_test is not None else testloader.dataset.data
    X_train = trainloader.dataset.data[:n_train] if n_train is not None else trainloader.dataset.data
    if float_tensor:
        X_test = torch.FloatTensor(X_test)
        X_train = torch.FloatTensor(X_train)
    return X_train, X_test