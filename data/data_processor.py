import numpy as np
import torch
from sklearn.model_selection import train_test_split

class Dataprocessor:
    """Class for data preprocessing operations"""

    @staticmethod
    def extract_limit(data, low=0.005, high=0.995):
        """Extract the lower and upper bounds for data clipping"""
        lower_bound = np.quantile(data, axis=1, q=low, method='nearest')
        upper_bound = np.quantile(data, axis=1, q=high, method='nearest')
        return lower_bound, upper_bound
    
    def extract_label(S, X, Y, labels=['CN', 'AD'], sex = 'A'): # 두번째 꺼(extract_label2)
        """Extract and process labels"""
        X2 = []
        Y2 = []

        for i in range(len(Y)):
            if Y[i] in labels:
                # Process sex information
                if sex == 'A' or S[i] == sex:
                    if X[i, 0] == 'M':
                        X2.append([0] + list(X[i,1:]))
                    elif X[i, 0] == 'F':
                        X2.append([1] + list(X[i,1:]))

                # Convert Labels to binary
                Y2.append(1 if labels.index(Y[i]) == 1 else 0)
            
        # Convert to numpy arrays and normalize
        X2 = np.array(X2).astype(float)
        Y2 = np.array(Y2)

        # Print statistics
        print(f"Total samples: {len(Y2)}")
        print(f"{labels[1]}: {sum(Y2)}")
        print(f"{labels[0]}: {len(Y2) - sum(Y2)}")

        # Standardize features
        mean =  np.mean(X2[:,1:])
        std = np.std(X2[:,1:])
        X2[:,1:] = (X2[:,1:] - mean) / std

        return X2, Y2, mean, std
    
    @staticmethod
    def gaussian(self, X): # train data에 대한 gaussian normalization
        """Gaussian normalization with outlier clipping"""
        X = X.T
        Zmin, Zmax = Dataprocessor.extract_limit(X)
        mean = np.expand_dims(np.mean(X, axis=1), axis=1)
        std = np.expand_dims(np.std(X, axis=1), axis=1)
        X = np.clip(X, Zmin, Zmax)
        X = (X - mean) / std
        X = X.T
        return X, mean, std
    
    @staticmethod
    def gaussian2(X, mean, std): # test data에 대한 gaussian normalization
        """Apply pre-computed gaussian noramlization"""
        X = X.T
        Zmin, Zmax = Dataprocessor.extract_limit(X)
        X = np.clip(X, Zmin, Zmax)
        X = (X - mean) / std
        X = X.T
        return X 

    def add_noise(data, noise_factor=0.1):
        """Add Gaussian noise to the input data"""
        noise = noise_factor * torch.randn_like(data)
        data[:, 1:] = data[:, 1:] + noise[:, 1:]
        return data

