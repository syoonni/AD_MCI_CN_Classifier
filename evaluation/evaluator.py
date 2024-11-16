import torch
import pandas as pd
import numpy as np
import torchmetrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torcheval.metrics.functional import multiclass_f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """Class for model evaluation"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize evaluation metrics"""
        return {
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=2).to(self.device),
            'precision': torchmetrics.Precision(task='multiclass', num_classes=2).to(self.device),
            'recall': torchmetrics.Recall(task='multiclass', num_classes=2).to(self.device),
            'specificity': torchmetrics.Specificity(task='multiclass', num_classes=2).to(self.device),
            'f1': torchmetrics.F1Score(task='multiclass', num_classes=2).to(self.device),
            'auroc': torchmetrics.AUROC(task='multiclass', num_classes=2).to(self.device)
        }
    
    def evaluate(self, test_loader, model_path = 'Object_best.pt'):
        """Evaluate the model on the data"""
        # Load model state
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict['State_dict'])  # 키 이름 변경
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)

                target = target.to(self.device, dtype=torch.int64)

                output = self.model(data)
                output = torch.softmax(output, dim=1)

                all_predictions.append(output)
                all_targets.append(target)

        # Concatenate all batches
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        results = self.calculate_and_save_metrics(predictions, targets)
        
        return results
    
    def calculate_and_save_metrics(self, predictions, targets):
        """Calculate metrics and save results"""
        results = {}

        targets = targets.long()
        
        # Calculate all metrics
        for name, metric in self.metrics.items():
            results[name] = metric(predictions, targets).item()

        # Save confusion matrix
        predictions_class = torch.argmax(predictions, dim=1)
        cm = confusion_matrix(
            targets.cpu().numpy(), 
            predictions_class.cpu().numpy(), 
            normalize='true'
        )
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('Confusion_matrix.png', dpi=300)
        plt.close()

        # Save detailed predictions
        df = pd.DataFrame({
            'True_Label': targets.cpu().numpy(),
            'Predicted_Label': predictions_class.cpu().numpy(),
            'Prob_Class_0': predictions[:, 0].cpu().numpy(),
            'Prob_Class_1': predictions[:, 1].cpu().numpy()
        })
        df.to_csv('predictions.csv', index=False)
        
        return results

class CombinedModelEvaluator(Evaluator):
    """Class for evaluating combined models"""
    def __init__(self, model1, model2, device, combine_method='threshold'):
        super().__init__(None, device)
        self.model1 = model1
        self.model2 = model2
        self.combine_method = combine_method

    def _get_predictions(self, dataloader):
        """Get predictions from combined models"""
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                # Validate data size
                if data.shape[1] < (self.model1.fc1.in_features + self.model2.fc1.in_features):
                    raise ValueError("Input data does not have enough features for both models.")

                # Split data for each model
                data1 = data[:, :self.model1.fc1.in_features]
                data2 = data[:, -self.model2.fc1.in_features:]

                # Get predictions from both models
                output1 = self.model1(data1)
                output2 = self.model2(data2)

                # Apply softmax to both outputs
                output1 = torch.softmax(output1, dim=1)
                output2 = torch.softmax(output2, dim=1)

                # Combine predictions
                combined_output = self._combine_predictions(output1, output2)

                predictions.append(combined_output)
                targets.append(target)

            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)

            return predictions, targets

    def _combine_predictions(self, pred1, pred2):
        """Combine predictions from two models"""
        if self.combine_method == 'average':
            return (pred1 + pred2) / 2
    
        elif self.combine_method == 'weighted':
            return 0.2 * pred1 + 0.8 * pred2
        elif self.combine_method == 'threshold':
            combined = torch.zeros_like(pred1)
            mask = torch.nn.Softmax(dim=1)(pred1)[:, 1] > 0.5
            combined[mask] = pred1[mask]
            combined[~mask] = 0.2 * pred1[~mask] + 0.8 * pred2[~mask]
            return combined
        else:
            raise ValueError(f"Unknown combine method: {self.combine_method}")
        
    def evaluate(self, test_loader, model_paths=None):
        """Evaluate combined models
        
        Args:
            test_loader: DataLoader for test data
            model_paths: Dictionary containing paths for both models
                        {'model1': path_to_model1, 'model2': path_to_model2}
        """
        # Load model weights if paths provided
        if model_paths:
            state_dict1 = torch.load(model_paths['model1'], map_location=self.device, weights_only=True)
            state_dict2 = torch.load(model_paths['model2'], map_location=self.device, weights_only=True)
            self.model1.load_state_dict(state_dict1['State_dict'])
            self.model2.load_state_dict(state_dict2['State_dict'])

        # Set models to evaluation mode
        self.model1.eval()
        self.model2.eval()

        # Get predictions using existing method
        predictions, targets = self._get_predictions(test_loader)

        # Calculate and save metrics
        results = self.calculate_and_save_metrics(predictions, targets)
        
        return results
