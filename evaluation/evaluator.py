import os
import torch
import pandas as pd
import numpy as np
import torchmetrics
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class Evaluator:
    """Class for model evaluation"""
    def __init__(self, model, device, model_name):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self):
        """Initialize evaluation metrics"""
        return {
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=2).to(self.device),
            'precision': torchmetrics.Precision(task='multiclass', num_classes=2, average='weighted').to(self.device),
            'recall': torchmetrics.Recall(task='multiclass', num_classes=2, average='weighted').to(self.device),
            'specificity': torchmetrics.Specificity(task='binary', num_classes=2, average='weighted').to(self.device),
            'f1': torchmetrics.F1Score(task='multiclass', num_classes=2, average='weighted').to(self.device),
            'auroc': torchmetrics.AUROC(task='multiclass', num_classes=2).to(self.device)
        }
    
    def evaluate(self, test_loader, model_path, output_dir):
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

        # Calculate and save metrics
        results = self.calculate_and_save_metrics(predictions, targets, self.model_name, output_dir)
        
        return results
    
    def calculate_and_save_metrics(self, predictions, targets, model_name, output_dir):
        """Calculate metrics and save results"""
        results = {}

        targets = targets.long()
        predictions_class = torch.argmax(predictions, dim=1)
        
        # Calculate metrics using raw probabilities for AUROC and class predictions for others
        for name, metric in self.metrics.items():
            if name == 'auroc':
                metric.update(predictions, targets)
            elif name == 'recall':
                cm = confusion_matrix(targets.cpu().numpy(), predictions_class.cpu().numpy())
                tn, fp, fn, tp = cm.ravel()
                manual_recall = tp / (tp + fn)
                results[name] = manual_recall  # Recall 계산 결과를 사용
            else:
                metric.update(predictions_class, targets)
                results[name] = metric.compute().item()

        # Save confusion matrix
        cm = confusion_matrix(
            targets.cpu().numpy(), 
            predictions_class.cpu().numpy(), 
            normalize='true'
        )
        tn, fp, fn, tp = cm.ravel()
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'), dpi=300)
        plt.close()

        # Save detailed predictions
        df = pd.DataFrame({
            'True_Label': targets.cpu().numpy(),
            'Predicted_Label': predictions_class.cpu().numpy(),
            'Prob_Class_0': predictions[:, 0].cpu().numpy(),
            'Prob_Class_1': predictions[:, 1].cpu().numpy()
        })
        df.to_csv(os.path.join(output_dir, f'predictions_{model_name}.csv'), index=False)

        recall_value = self.metrics['recall'].compute().item()
        print(f"Updated Recall: {recall_value}")

        # Manual Recall 계산 결과
        manual_recall = tp / (tp + fn)
        print(f"Manual Recall: {manual_recall}")

        return results
    

class CombinedModelEvaluator(Evaluator):
    """Class for evaluating combined models"""
    def __init__(self, model1, model2, device, combine_method='threshold'):
        super().__init__(None, device, model_name='combined')
        self.model1 = model1
        self.model2 = model2
        self.combine_method = combine_method

    def _get_predictions(self, testloader1, testloader2):
        """Get predictions from combined models"""
        predictions = []
        targets = []

        with torch.no_grad():
            for (data1, target1), (data2, target2) in zip(testloader1, testloader2):
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)
                target1 = target1.to(self.device, dtype=torch.int64)

                # Get predictions from both models
                output1 = self.model1(data1)
                output2 = self.model2(data2)

                # Apply softmax to both outputs
                output1 = torch.softmax(output1, dim=1)
                output2 = torch.softmax(output2, dim=1)

                # Combine predictions
                combined_output = self._combine_predictions(output1, output2)

                predictions.append(combined_output)
                targets.append(target1)           

            predictions = torch.cat(predictions, dim=0)
            targets = torch.cat(targets, dim=0)

            return predictions, targets

    def _combine_predictions(self, pred1, pred2):
        """Combine predictions from two models"""
        if self.combine_method == 'average':
            return (pred1 + pred2) / 2
    
        elif self.combine_method == 'weighted':
            return 0.7 * pred1 + 0.3 * pred2
        elif self.combine_method == 'threshold':
            combined = torch.zeros_like(pred1)
            mask = torch.nn.Softmax(dim=1)(pred1)[:, 1] > 0.5
            combined[mask] = pred1[mask]
            combined[~mask] = 0.2 * pred1[~mask] + 0.8 * pred2[~mask]
            return combined
        else:
            raise ValueError(f"Unknown combine method: {self.combine_method}")
        
    def evaluate(self, test_loader1, testloader2, model_paths, output_dir):
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
        predictions, targets = self._get_predictions(test_loader1, testloader2)

        # Calculate and save metrics
        results = self.calculate_and_save_metrics(predictions, targets, 'combined', output_dir)
        
        return results
