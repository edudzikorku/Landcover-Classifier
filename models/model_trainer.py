import numpy as np # type: ignore
from typing import Dict, Any
from .model_io import ModelIO
from sklearn.preprocessing import LabelEncoder  # type: ignore
from imblearn.over_sampling import SMOTE, ADASYN  # type: ignore
from imblearn.combine import SMOTETomek, SMOTEENN  # type: ignore
from imblearn.over_sampling import BorderlineSMOTE  # type: ignore
from evaluation import ModelEvaluator
from evaluation import ModelVisualizer
from sklearn.metrics import accuracy_score  # type: ignore
from config import CSV_PATH, PLOTS_PATH, MODELS_PATH
from sklearn.impute import SimpleImputer # type: ignore


class BalancingTechniques:
    """
    Class balancing techniques for imbalanced datasets.
    Useful for avoiding model bias towards majority class.
    """
    
    @staticmethod
    def check_for_nans(X: np.ndarray, y: np.ndarray) -> bool:
        """Check if target array contains NaN values"""
        return np.isnan(X).any() or np.isnan(y).any()
    
    @staticmethod
    def imput_missing_values(X: np.ndarray, strategy: str = 'median') -> np.ndarray:
        """Impute missing values in feature matrix"""
        if strategy == 'zero':
            X = np.nan_to_num(X, nan = 0.0)
        else:
            imputer = SimpleImputer(strategy = strategy)
            X = imputer.fit_transform(X)
        return X
    
    @staticmethod
    def apply_balancing(X: np.ndarray, y: np.ndarray, technique: str = 'smote', 
                        imputation_strategy: str = 'median', **kwargs) -> (np.ndarray, np.ndarray):
        """
        Apply class balancing technique
        
        Args:
            - X: Feature matrix
            - y: Target vector
            - technique: Balancing technique ('none', 'smote', 'adasyn', 'bsmote', 'smote_tomek', 'smote_enn')
            - imputation_strategy: Strategy to handle missing values before balancing 
                                ('none', 'smote', 'adasyn', 'bsmote', 'smote_tomek', 'smote_enn'). Default is 'median'.
            **kwargs: Additional parameters for the balancing technique
            
        Returns:
            Tuple of balanced X and y
        """
        if technique == 'none':
            return X, y

        # Check for NaNs and impute if necessary
        if BalancingTechniques.check_for_nans(X, y):
            print(f'Warning: Missing values detected. Imputing {imputation_strategy} before {technique}...')   
            X = BalancingTechniques.imput_missing_values(X, strategy = imputation_strategy)
        # Remove any nans in y 
        nan_mask = ~np.isnan(y)
        X = X[nan_mask]
        y = y[nan_mask]

        random_state = kwargs.pop('random_state', 42)
       
        if technique == 'smote':
            balancer = SMOTE(random_state = random_state, **kwargs)
            
        elif technique == 'adasyn':
            balancer = ADASYN(random_state = random_state, **kwargs)
            
        elif technique == 'bsmote':
            # BorderlineSMOTE
            balancer = BorderlineSMOTE(random_state = random_state, **kwargs)
            
        elif technique == 'smote_tomek':
            # SMOTE + Tomek links for cleaning
            balancer = SMOTETomek(random_state = random_state, **kwargs)
            
        elif technique == 'smote_enn':
            # SMOTE + Edited Nearest Neighbors for cleaning
            balancer = SMOTEENN(random_state = random_state, **kwargs)
            
        else:
            raise ValueError(f"Unknown balancing technique: {technique}")
        
        return balancer.fit_resample(X, y)


def train_ml_model(model, X_train, X_test, y_train, y_test, feature_names, 
                  balancing_technique: str = 'None', 
                  imputation_strategy: str = 'median',
                  **balancing_kwargs) -> Dict[str, Any]:
    """
    Train and evaluate a ML model using separated components

    Args:
        - model: An instance of a machine learning model (e.g., RandomForest, XGBoost).
        - X_train (np.ndarray): Training feature matrix.
        - X_test (np.ndarray): Testing feature matrix.
        - y_train (np.ndarray): Training target vector.
        - y_test (np.ndarray): Testing target vector.
        - feature_names (List[str]): List of feature names.
        - balancing_technique (str): Class balancing technique to apply
        - imputation_strategy (str): Imputation strategy to apply before balancing. Default is 'median'.
        - balancing_kwargs: Additional arguments for balancing technique

    Returns:
        - Dict[str, Any]: A dictionary containing the test accuracy and cross-validation results.
    """
    print(f"Using balancing: {balancing_technique}")
    print(f"Using imputation: {imputation_strategy}")
    
    if balancing_kwargs:
        print(f"Balancing parameters: {balancing_kwargs}")
    
    # Preprocess data
    X_train_scaled = model.preprocess_data(X_train)
    X_test_scaled = model.scaler.transform(X_test)

    # Handle nan values in test data 
    if np.isnan(X_test_scaled).any():
        print(f'Warning: Missing values detected in test data. Imputing {imputation_strategy}...')
        # imputer = SimpleImputer(strategy = imputation_strategy)
        X_test_scaled = BalancingTechniques.imput_missing_values(X_test_scaled, strategy = imputation_strategy)

    # Handle label encoding if needed
    if model.needs_label_encoding():
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    # Apply class balancing if specified
    if balancing_technique != 'none':
        print(f"Applying {balancing_technique} balancing...")
        print(f"Original class distribution: {np.bincount(y_train)}")
        
        X_train_scaled, y_train = BalancingTechniques.apply_balancing(
            X_train_scaled, y_train, 
            technique = balancing_technique, 
            imputation_strategy = imputation_strategy,
            **balancing_kwargs
        )
        
        print(f"Balanced class distribution: {np.bincount(y_train)}")

    # Tune hyperparameters
    model.tune_hyperparameters(X_train_scaled, y_train)

    # Create evaluator and visualizer
    evaluator = ModelEvaluator(model)
    visualizer = ModelVisualizer(model, evaluator)
    
    # Perform cross-validation
    cv_results = evaluator.perform_cross_validation(X_train_scaled, y_train, feature_names)

    # Create visualizations
    visualizer.plot_accuracies(X_train_scaled, y_train, save_path = PLOTS_PATH)  # type: ignore
    if cv_results['feature_importance_folds'] is not None:
        visualizer.plot_feature_importance(cv_results['feature_importance_folds'], PLOTS_PATH)  # type: ignore

    # Train final model on full training data
    model.fit(X_train_scaled, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Save results using ModelIO
    ModelIO.save_model(model, MODELS_PATH)  # type: ignore
    ModelIO.save_cv_results(evaluator, CSV_PATH)  # type: ignore
    ModelIO.save_fold_metrics_to_dataframe(evaluator, CSV_PATH)  # type: ignore
    ModelIO.save_classification_report(y_test, y_pred, 
                                      f"{model.__class__.__name__}_{balancing_technique}", 
                                      CSV_PATH)  # type: ignore

    return {
        'test_accuracy': test_accuracy,
        'cv_results': cv_results,
        'balancing_technique': balancing_technique
    }