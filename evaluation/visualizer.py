import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
from .evaluator import ModelEvaluator
from models import BaseModel
from config import DEFAULT_PLOT_STYLE, EXPORT_RESOLUTION, FIGURE_SIZE


plt.style.use(DEFAULT_PLOT_STYLE)

class ModelVisualizer:
    """Handles all visualisation tasks for model analysis"""
    
    def __init__(self, model: BaseModel, evaluator: ModelEvaluator):
        self.model = model
        self.evaluator = evaluator

    def plot_feature_importance(self, feature_importance: pd.DataFrame, save_path: str) -> None:
        """
        Plot feature importance

        Args:
            - feature_importance (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
            - save_path (str): Directory to save the plot.

        Returns:
            - None
        """
        plt.figure(figsize = FIGURE_SIZE)
        feature_importance.sort_values(by = 'importance', ascending = False).plot(
            kind = 'barh', x = 'feature', y = 'importance',
            title = f"Feature Importance - {self.model.__class__.__name__}"
        )
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f"{save_path}/{self.model.__class__.__name__}_feature_importance.png", 
                   dpi = EXPORT_RESOLUTION, bbox_inches = 'tight')
        plt.close()

    def plot_accuracies(self, X: np.ndarray, y: np.ndarray, save_path: str) -> None:
        """
        Plot training and validation accuracies across folds

        Args:
            - X (np.ndarray): Feature matrix.
            = y (np.ndarray): Target vector.
            - save_path (str): Directory to save the plot.

        Returns:
            - None
        """
        if not self.evaluator.cv_results:
            raise ValueError("No cross-validation results available. Run cross-validation first.")
        
        fold_metrics = self.evaluator.cv_results['fold_metrics']
        train_accuracies = [fold['train_accuracy'] for fold in fold_metrics]
        val_accuracies = [fold['accuracy'] for fold in fold_metrics]
        
        plt.figure(figsize = FIGURE_SIZE)
        folds = range(1, len(fold_metrics) + 1)
        
        plt.plot(folds, train_accuracies, 'b-', label = 'Training Accuracy', marker = 'o')
        plt.plot(folds, val_accuracies, 'r-', label = 'Validation Accuracy', marker = 'o')
        
        # Add value annotations
        for fold, (train_acc, val_acc) in enumerate(zip(train_accuracies, val_accuracies), 1):
            plt.annotate(f'{train_acc:.3f}', (fold, train_acc), 
                        textcoords = "offset points", xytext = (0, 10), ha = 'center')
            plt.annotate(f'{val_acc:.3f}', (fold, val_acc),
                        textcoords = "offset points", xytext = (0, -15), ha = 'center')
        
        plt.title(f'Training vs Validation Accuracy - {self.model.__class__.__name__}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle = '--', alpha = 0.7)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/{self.model.__class__.__name__}_accuracies.png", 
                   dpi = EXPORT_RESOLUTION, bbox_inches = 'tight')
        plt.close()
