from typing import Dict, List, Any 
from .base_model import BaseModel 
from sklearn.ensemble import RandomForestClassifier # type: ignore 


class RFClassificationModel(BaseModel):
    """Random Forest classification implementation. Inherits from BaseModel."""

    def __init__(self, random_state: int = 42, n_splits: int = 10) -> None:
        super().__init__(random_state, n_splits = n_splits)
        self.model = RandomForestClassifier(random_state = random_state)

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get the hyperparameter grid for the model"""

        return {
              'n_estimators': [10, 20],
              'max_depth': [1, 2, 3, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['sqrt', 'log2', None],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy'],
              'max_leaf_nodes': [None, 10, 20, 30],
              'class_weight': [None, 'balanced']
        }
    
    def needs_label_encoding(self) -> bool:
        """Whether this model requires label-encoded targets"""
        return super().needs_label_encoding()
