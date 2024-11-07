import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import AlzheimerDataset
from network.regressor import Regressor
from training.trainer import Trainer
from evaluation.evaluator import Evaluator, CombinedModelEvaluator

def set_device():
    """Setup computation device and seed"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(777)
    if device == 'cuda':
        torch.cuda.manual_seed_all(777)
    return device

def set_model(in_dim, device, lr=0.001):
    """Setup the model"""
    model = Regressor(in_dim=in_dim, out_dim=2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=10, 
        min_lr=1e-10, 
        verbose=True
    )
    
    return model, criterion, optimizer, scheduler

def main():
    # Setup
    device = set_device()
    
    # Configuration
    labels = ['MCI', 'AD']
    sex = 'A'

    # First model (manual features)
    interesting_features1 = ['37', '13', '14']
    class1 = len(interesting_features1)
    dataset1 = AlzheimerDataset(
        data_path='path/to/your/data.csv',
        interesting_features=interesting_features1,
        labels=labels,
        sex=sex
    )

    dataloaders1 = dataset1.data_loaders()

    # Setup first model
    model1 = Regressor(len(interesting_features1), 2).to(device)
    criterion1 = nn.CrossEntropyLoss().to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1,
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=1e-10,
        verbose=True
    )

    # Train first model
    trainer1 = Trainer(model1, criterion1, optimizer1, scheduler1, device)
    trainer1.train(
        train_loader=dataloaders1['train'],
        val_loader=dataloaders1['val'],
        num_epochs=500,
        best_model_save_path='Object_best0.pt',
        final_model_save_path='Object_last0.pt'
    )

    # Evaluate first model
    evaluator1 = Evaluator(model1, device)
    results1 = evaluator1.evaluate(dataloaders1['test'], model_path='Object_last0.pt')
    print("\nFirst Model Results:")
    for metric_name, value in results1.items():
        print(f"{metric_name}: {value:.4f}")

    # Second model (Fast features)
    interesting_features2 = ['13', '14', '29', '28', '40', '46', '47', '38', '71', '39', '21', '3', '27']
    class2 = len(interesting_features2)

    dataset2 = AlzheimerDataset(
        data_path='path/to/your/data.csv',
        interesting_features=interesting_features2,
        labels=labels,
        sex=sex
    )
    dataloaders2 = dataset2.data_loaders()

    # Setup second model
    model2 = Regressor(len(interesting_features2), 2).to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2,
        mode='min',
        factor=0.1,
        patience=10,
        min_lr=1e-10,
        verbose=True
    )

    # Train second model
    trainer2 = Trainer(model2, criterion2, optimizer2, scheduler2, device)
    trainer2.train(
        train_loader=dataloaders2['train'],
        val_loader=dataloaders2['val'],
        num_epochs=500,
        best_model_save_path='Object_best1.pt',
        final_model_save_path='Object_last1.pt'
    )

    # Evaluate second model
    evaluator2 = Evaluator(model2, device)
    results2 = evaluator2.evaluate(dataloaders2['test'], model_path='Object_last1.pt')
    print("\nSecond Model Results:")
    for metric_name, value in results2.items():
        print(f"{metric_name}: {value:.4f}")

    ###### Combined Model ######
    intersting_combined_features = ['37', '13', '14', '29', '28', '40', '46', '47', '38', '71', '39', '21', '3', '27']

    dataset_combined = AlzheimerDataset(
        data_path='path/to/your/data.csv',
        interesting_features=intersting_combined_features,
        labels=labels,
        sex=sex
    )
    dataloaders_combined = dataset_combined.data_loaders()

    # Load model states for combined evaluation
    model1.load_state_dict(torch.load('Object_last0.pt', map_location=device)['State_dict'])
    model2.load_state_dict(torch.load('Object_last1.pt', map_location=device)['State_dict'])

    # Evaluate combined model
    combined_evaluator = CombinedModelEvaluator(model1, model2, device, combine_method='average')
    results_combined = combined_evaluator.evaluate(dataloaders_combined['test'])
    print("\nCombined Model Results:")
    for metric_name, value in results_combined.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()