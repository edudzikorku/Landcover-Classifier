from typing import Dict, List, Any 
from .base_model import BaseModel 
from catboost import CatBoostClassifier  # type: ignore


class CatBoostClassificationModel(BaseModel):
    """
    CatBoost classification algorithm implementation. Inherits from BaseModel.
       
    """
    def __init__(self, random_state: int = 42, n_splits: int = 10) -> None:
        super().__init__(random_state, n_splits = n_splits)
        self.model = CatBoostClassifier(random_state = random_state, verbose = 0, allow_writing_files = False, 
                                        loss_function = 'MultiClass')

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get the hyperparameter grid for the model"""

        return {
            'iterations': [10, 20],
            'learning_rate': [0.01, 0.1, 0.3],
            'depth': [3, 5, 7, 9],
            'l2_leaf_reg': [1, 3, 5],
            'border_count': [32, 64, 128],
            'bagging_temperature': [0, 0.5, 1],
            'random_strength': [0, 0.5, 1]
            
        }
    
    def needs_label_encoding(self) -> bool: 
        """Whether this model requires label-encoded targets"""
        return super().needs_label_encoding()
