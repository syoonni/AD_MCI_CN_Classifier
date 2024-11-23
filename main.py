import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

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
    
    ################## Configuration ##################
    labels = ['AD', 'CN']
    sex = 'A'
    num_epochs = 500

    # Create directories for models and results
    model_dir = 'models/models1/'
    results_dir = 'results/results1/'

    data_path1 = 'AlzheimerDataset/fast_data + manual.csv'
    data_path2 = 'AlzheimerDataset/fast_data + manual.csv'

    # First model (manual features + Fast features)
    interesting_features1 = ['manu_37', '13', '14', '29', '28', '40', 'Sum']

    # Second model (Fast features)
    interesting_features2 = ['13', '14', '29', '28', '40', '46', '47', '38', '71', '39', '21', '3', '27']

    ###################################################

    # Create directories if they don't exist
    for directory in [model_dir, results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    df_data1 = pd.read_csv(data_path1)
    df_data2 = pd.read_csv(data_path2)
    common_ids = set(df_data1['id']).intersection(set(df_data2['id']))

    dataset1 = AlzheimerDataset(
        data_path=data_path1,
        interesting_features=interesting_features1,
        labels=labels,
        sex=sex,
        common_ids=common_ids
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
        num_epochs=num_epochs,
        best_model_save_path=os.path.join(model_dir, 'Object_best1.pt'),
        final_model_save_path=os.path.join(model_dir, 'Object_last1.pt'),
        output_dir=results_dir
    )

    # First model evaluation
    evaluator1 = Evaluator(model1, device, "model1")
    results1 = evaluator1.evaluate(
        dataloaders1['test'],
        model_path=os.path.join(model_dir, 'Object_best1.pt'),
        output_dir=results_dir
    )


    dataset2 = AlzheimerDataset(
        data_path=data_path2,
        interesting_features=interesting_features2,
        labels=labels,
        sex=sex,
        common_ids=common_ids
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
        num_epochs=num_epochs,
        best_model_save_path=os.path.join(model_dir, 'Object_best2.pt'),
        final_model_save_path=os.path.join(model_dir, 'Object_last2.pt'),
        output_dir=results_dir
    )

    # Second model evaluation
    evaluator2 = Evaluator(model2, device, "model2")
    results2 = evaluator2.evaluate(
        dataloaders2['test'],
        model_path=os.path.join(model_dir, 'Object_best2.pt'),
        output_dir=results_dir
    )

    ####################### Combined Model ###########################

    # Evaluate combined model
    combined_evaluator = CombinedModelEvaluator(model1, model2, device, combine_method='average')
    results_combined = combined_evaluator.evaluate(
        dataloaders1['test'],
        dataloaders2['test'],
        model_paths={  
            'model1': os.path.join(model_dir, 'Object_best1.pt'),
            'model2': os.path.join(model_dir, 'Object_best2.pt')
        },
        output_dir=results_dir
    )

    with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
        for model_name, results in [
            ("First Model", results1),
            ("Second Model", results2),
            ("Combined Model", results_combined)
        ]:
            # Print and save results
            print(f"\n{model_name} Results:")
            f.write(f"\n{model_name} Results:\n")

            for metric_name, value in results.items():
                print(f"{metric_name}: {value:.4f}")
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()