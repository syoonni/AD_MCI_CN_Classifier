from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.data_processor import Dataprocessor
from torch.utils.data import DataLoader

class TensorData(Dataset):
    """Dataset class for handling the Alzheimer's data"""
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
class AlzheimerDataset:
    """Class for handling Alzheimer's dataset loading and preprocessing"""

    def __init__(self, data_path, interesting_features, labels=['CN', 'AD'], sex='A', test = False, mean = None, std = None):
        """Initialize dataset with configuration"""
        self.data_path = data_path
        self.interesting_features = interesting_features
        self.labels = labels
        self.sex = sex
        self.test = test
        self.mean = mean
        self.std = std

        self.load_and_split_data()
        
    def load_and_split_data(self):
        """Load data and split into train, validation and test sets"""
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Extract relevant columns
        S = df['Sex'].values
        X = df[[str(i) for i in self.interesting_features]].values
        Y = df['label'].values
        
        # Process data
        X, Y = Dataprocessor.extract_label(S, X, Y, labels = self.labels, sex = self.sex)

        if self.test:
            # For test data, use provided mean and std
            X = Dataprocessor.gaussian2(X, self.mean, self.std)
            self.X_test = X
            self.Y_test = Y
        
        else:
            # For training data, compute mean and std
            X, mean, std = Dataprocessor.gaussian(X)
            self.mean = mean
            self.std = std
        
            # Split data
            X_train_val, X_test, Y_train_val, Y_test = train_test_split(
                X, Y, test_size=0.3, shuffle=False
            )
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train_val, Y_train_val, test_size=0.1, shuffle=False
            )
            
            return {
                'train': (X_train, Y_train),
                'val': (X_val, Y_val),
                'test': (X_test, Y_test)
            }
    
    def data_loaders(self, batch_size=32):
        """Create DataLoader objects for all splits"""
        data_splits = self.load_and_split_data()
        
        dataloaders = {}
        for split_name, (X, Y) in data_splits.items():
            dataset = TensorData(X, Y)
            shuffle = split_name == 'train'  # Only shuffle training data
            dataloaders[split_name] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
            )
            
        return dataloaders
    