import numpy as np
import pandas as pd # type: ignore
from typing import Dict, Any, List
from models import BaseModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore

class ModelEvaluator:
    """Handles model evaluation, cross-validation, and analysis"""
    
    def __init__(self, model: BaseModel):
        self.model = model
        self.cv_results = None
        self.shap_values = None

    def perform_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Perform cross-validation and collect detailed metrics

        Args:
            - X (np.ndarray): Feature matrix.
            - y (np.ndarray): Target vector.
            - feature_names (List[str]): List of predictors.
        """
        fold_metrics = []
        feature_importance_folds = []
        train_accuracies = []
        precisions, recalls, f1_scores = [], [], []

        for fold, (train_idx, val_idx) in enumerate(self.model.kfold.split(X), 1):
            # Split data for this fold
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Train the model on this fold
            self.model.model.fit(X_train_fold, y_train_fold) # type: ignore

            # Calculate metrics
            y_train_pred = self.model.model.predict(X_train_fold) # type: ignore
            y_val_pred = self.model.model.predict(X_val_fold) # type: ignore
            
            fold_train_accuracy = accuracy_score(y_train_fold, y_train_pred)
            fold_accuracy = accuracy_score(y_val_fold, y_val_pred)
            
            train_accuracies.append(fold_train_accuracy)
            
            # Classification report for detailed metrics
            fold_report = classification_report(y_val_fold, y_val_pred, output_dict=True)
            class_metrics = {k: v for k, v in fold_report.items() 
                           if k not in ['accuracy', 'macro avg', 'weighted avg']}
            
            fold_precision = np.mean([metrics['precision'] for metrics in class_metrics.values()])
            fold_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
            fold_f1 = np.mean([metrics['f1-score'] for metrics in class_metrics.values()])
            
            precisions.append(fold_precision)
            recalls.append(fold_recall)
            f1_scores.append(fold_f1)

            # Feature importance if available
            if hasattr(self.model.model, "feature_importances_"):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.model.model.feature_importances_ # type: ignore
                })
                feature_importance_folds.append(feature_importance)

            fold_metrics.append({
                'fold': fold,
                'train_accuracy': fold_train_accuracy,
                'accuracy': fold_accuracy,
                'precision': fold_precision,
                'recall': fold_recall,
                'f1_score': fold_f1,
                'classification_report': fold_report
            })

        # Calculate averages and standard deviations
        self.cv_results = { # type: ignore
            'fold_metrics': fold_metrics,
            'avg_train_accuracy': np.mean(train_accuracies),
            'std_train_accuracy': np.std(train_accuracies),
            'avg_accuracy': np.mean([fold['accuracy'] for fold in fold_metrics]),
            'std_accuracy': np.std([fold['accuracy'] for fold in fold_metrics]),
            'precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'f1_score': np.mean(f1_scores),
            'std_f1_score': np.std(f1_scores),
            'feature_importance_folds': pd.concat(feature_importance_folds).groupby('feature').mean().reset_index() if feature_importance_folds else None
        }
        
        return self.cv_results # type: ignore

    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Generate classification report

        Args:
            - y_true (np.ndarray): True labels.
            - y_pred (np.ndarray): Predicted labels.
        """
        return classification_report(y_true, y_pred, output_dict = True)

    def get_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix

        Args:
            - y_true (np.ndarray): True labels.
            - y_pred (np.ndarray): Predicted labels.
        """
        return confusion_matrix(y_true, y_pred)