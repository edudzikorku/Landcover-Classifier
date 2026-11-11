import os
import joblib # type: ignore
import pandas as pd # type: ignore
from .base_model import BaseModel
from evaluation import ModelEvaluator
from sklearn.metrics import classification_report, confusion_matrix # type: ignore

class ModelIO:
    """Handles saving and loading of models and results"""
    
    @staticmethod
    def save_model(model: BaseModel, save_path: str) -> None:
        """
        Save the trained model and scaler

        Args:
            - model (BaseModel): The trained model instance.
            - save_path (str): Directory path to save the model files.

        Returns:
            - None
        """
        model_name = model.__class__.__name__
        joblib.dump(model.model, f"{save_path}/{model_name}.joblib")
        joblib.dump(model.scaler, f"{save_path}/{model_name}_scaler.joblib")

    @staticmethod
    def save_cv_results(evaluator: ModelEvaluator, save_path: str) -> None:
        """
        Save cross-validation results

        Args:
            - evaluator (ModelEvaluator): The evaluator instance containing CV results.
            - save_path (str): Directory path to save the CV results.
            
        Returns:
            - None
        """
        model_name = evaluator.model.__class__.__name__
        joblib.dump(evaluator.cv_results, f"{save_path}/{model_name}_cv_results.joblib")

    @staticmethod
    def save_fold_metrics_to_dataframe(evaluator: ModelEvaluator, save_path: str) -> pd.DataFrame:
        """
        Save fold metrics to CSV
        Args:

            - evaluator (ModelEvaluator): The evaluator instance containing fold metrics.
            - save_path (str): Directory path to save the fold metrics CSV.

        Returns:
        - pd.DataFrame: DataFrame containing fold metrics.

        """
        if not evaluator.cv_results:
            raise ValueError("No cross-validation results available")
        
        fold_data = []
        model_name = evaluator.model.__class__.__name__
        
        for fold_metric in evaluator.cv_results['fold_metrics']:
            fold_data.append({
                'Model': model_name,
                'Fold': fold_metric['fold'],
                'Training Accuracy': fold_metric['train_accuracy'],
                'Validation Accuracy': fold_metric['accuracy'],
                'Precision': fold_metric['precision'],
                'Recall': fold_metric['recall'],
                'F1-Score': fold_metric['f1_score']
            })
        
        fold_df = pd.DataFrame(fold_data)
        fold_df = fold_df.sort_values(by = 'Fold')
        fold_df.to_csv(f"{save_path}/{model_name}_fold_metrics.csv", index = False)
        return fold_df

    @staticmethod
    def save_classification_report(y_true, y_pred, model_name: str, save_path: str) -> None:
        """
        Save classification report and confusion matrix

        Args:
            - y_true (np.ndarray): True labels.
            - y_pred (np.ndarray): Predicted labels.
            - model_name (str): Name of the model.
            - save_path (str): Directory path to save the classification report and confusion matrix.
            
        Returns:
            - None
        """
        # Classification report
        report_dict = classification_report(y_true, y_pred, output_dict = True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(save_path, f"{model_name}_classification_report.csv"))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Assuming you have access to class labels - you might need to pass this as parameter
        cm_df = pd.DataFrame(cm)
        cm_df.to_csv(os.path.join(save_path, f"{model_name}_confusion_matrix.csv"))
