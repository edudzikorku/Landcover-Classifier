import os
import joblib
import traceback
import numpy as np # type: ignore
import config
from datetime import datetime as dt
import pandas as pd # type: ignore
from models import train_ml_model
from sklearn.model_selection import train_test_split    # type: ignore
from utils import DataProcessor
from models import RFClassificationModel, XGBClassificationModel, CatBoostClassificationModel

import warnings
warnings.filterwarnings("ignore")

def main():

    # Initialize DataProcessor
    data_processor = DataProcessor(config.RASTER_PATH, config.SHP_PATH)

    # Create necessary directories
    data_processor.create_directories(config.DIR_LIST)

    # Extract LULC points from each polygon
    raster_gdf = data_processor.extract_points_by_lulc(config.LULC_SPECIFICATIONS)

    # # Extract pixel values from raster
    raster_gdf = data_processor.extract_polygon_values()

    raster_gdf = raster_gdf[[col for col in raster_gdf.columns.unique() if col not in config.FIELDS_TO_REMOVE]]

    # #  Filter out rows where all specified columns are 0.0 or -0.0
    cols = [col for col in raster_gdf.columns.unique()]
    # # Ensure the 'Value' column is numeric
    raster_gdf[config.TARGET_COLUMN] = pd.to_numeric(raster_gdf[config.TARGET_COLUMN], errors = 'coerce')
    raster_gdf_mask = raster_gdf[~(raster_gdf[cols].abs() < config.TOLERANCE).any(axis = 1)]

    feature_columns = [col for col in raster_gdf_mask.columns.unique() if col not in config.TARGET_COLUMN] # All
    print(f"Feature columns: {feature_columns}")


    gdf, _, _, X, y = data_processor.get_features_and_targets(feature_columns = feature_columns,
                                                          target_column = config.TARGET_COLUMN,
                                                          imputation_strategy = 'median',
                                                          geodataframe = raster_gdf_mask)

    # Create initial train/test split indices
    train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size = config.TEST_SIZE,
                random_state = config.RANDOM_STATE,
                stratify = y
            )

    # Prepare data for traditional ML models
    X_train_ml = X[train_idx]
    X_test_ml = X[test_idx]
    y_train_ml = y[train_idx]
    y_test_ml = y[test_idx]


    # Train traditional ml models
    ml_model_list = [
        RFClassificationModel(n_splits = config.N_SPLITS),
        XGBClassificationModel(n_splits = config.N_SPLITS),
        CatBoostClassificationModel(n_splits = config.N_SPLITS)
    ]

    all_results = {}


    # Train and evaluate each ML model
    for model in ml_model_list:
        model_name = model.__class__.__name__
        print(f"\n{'='*60}")
        print(f"Training {model_name} model with {config.BALANCING_TECHNIQUE} balancing...")
        print(f"Start time: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        try:
            # Extract balancing parameters from config
            balancing_params = config.BALANCING_PARAMS.copy()
             # Add technique-specific parameters
            if config.BALANCING_TECHNIQUE == 'smote':
                # Only pass relevant parameters for SMOTE
                relevant_params = {k: v for k, v in balancing_params.items() 
                                 if k in ['k_neighbors', 'random_state']}
            elif config.BALANCING_TECHNIQUE == 'bsmote':
                # BorderlineSMOTE uses both k_neighbors and m_neighbors
                relevant_params = {k: v for k, v in balancing_params.items() 
                                 if k in ['k_neighbors', 'm_neighbors', 'random_state']}
            elif config.BALANCING_TECHNIQUE in ['adasyn', 'smote_tomek', 'smote_enn']:
                # These might use k_neighbors
                relevant_params = {k: v for k, v in balancing_params.items() 
                                 if k in ['k_neighbors', 'random_state']}
            else:
                # 'none' technique - no parameters needed
                relevant_params = {}

            results = train_ml_model(
                    model,  
                    X_train_ml, 
                    X_test_ml, 
                    y_train_ml, 
                    y_test_ml, 
                    feature_columns,
                    balancing_technique = config.BALANCING_TECHNIQUE,
                    imputation_strategy = config.IMPUTATION_STRATEGY,
                    **relevant_params
                )
                
           # Add balancing technique info to results
            results['balancing_technique'] = config.BALANCING_TECHNIQUE
            results['imputation_strategy'] = config.IMPUTATION_STRATEGY
            
            all_results[model_name] = results
            print(f"Completed {model_name} with {config.BALANCING_TECHNIQUE}")
                
        except Exception as e:
           print(f"Failed to train {model_name}: {str(e)}")
           traceback.print_exc()

    # Create summary dataframe
    summary_data = []
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'Balancing Technique': results.get('balancing_technique', 'none'),
            'Imputation Strategy': results.get('imputation_strategy', 'none'),
            'Test Accuracy': results['test_accuracy']
        }

        if 'cv_results' in results:
            row.update({
                'CV Accuracy': results['cv_results']['avg_accuracy'],
                'CV Std': results['cv_results']['std_accuracy'],
                'Avg-Recall': results['cv_results']['recall'],
                'Recall Std': results['cv_results']['std_recall'],
                'Avg-Precision': results['cv_results']['precision'],
                'Precision Std': results['cv_results']['std_precision'],
                'Avg-F1': results['cv_results']['f1_score'],
                'F1 Std': results['cv_results']['std_f1_score'],
                'Average Training Accuracy': results['cv_results']['avg_train_accuracy'],
                'Std of Training Accuracy': results['cv_results']['std_train_accuracy']
            })

        summary_data.append(row)

     # Save summary dataframe
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        os.path.join(config.CSV_PATH, f'model_comparison_{config.BALANCING_TECHNIQUE}_{timestamp}.csv'), 
        index = False
    )
    
    print("\n" + "="*60)
    print("Model comparison summary:")
    print(f"Balancing Technique: {config.BALANCING_TECHNIQUE}")
    print(f"Imputation Strategy: {config.IMPUTATION_STRATEGY}")
    print("-"*60)
    print(f"\n{summary_df.to_string()}")
    
    # Also save a detailed results file
    detailed_results = {
        'config': {
            'balancing_technique': config.BALANCING_TECHNIQUE,
            'imputation_strategy': config.IMPUTATION_STRATEGY,
            'balancing_params': config.BALANCING_PARAMS
        },
        'results': all_results
    }

    joblib.dump(
        detailed_results, 
        os.path.join(config.CSV_PATH, f'detailed_results_{config.BALANCING_TECHNIQUE}_{timestamp}.joblib')
    )
    
    print("\nTraining completed!")
    print(f"End time: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()