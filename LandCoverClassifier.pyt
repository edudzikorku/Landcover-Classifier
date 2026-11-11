# LandCoverClassifier.pyt
"""
ArcGIS Pro Toolbox for Landcover Classification
Supports Random Forest, XGBoost, and CatBoost models with flexible
class balancing and preprocessing options.
"""
import time
import arcpy
import sys
import os
import json
import traceback
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
from datetime import timedelta
from datetime import datetime as dt

# Set GDAL environment variable BEFORE any rasterio imports
os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'

import rasterio as rio
from sklearn.model_selection import train_test_split

class Toolbox(object):
    """Landcover Classifier Toolbox"""
    
    def __init__(self):
        self.label = "Landcover Classifier"
        self.alias = "LandcoverClassifier"
        self.description = """
        Toolbox for landcover classification using machine learning algorithms.
        Supports Random Forest, XGBoost, and CatBoost models with flexible
        class balancing and preprocessing options.
        """
        self.tools = [
            TrainLandcoverModel,
            PredictLandcover,
            SpectralSeparabilityAnalysis,
            ApplyMajorityFilter
        ]

class TrainLandcoverModel(object):
    """Train machine learning models for landcover classification."""
    
    def __init__(self):
        self.label = "1. Train Landcover Model"
        self.description = """
        Train machine learning models for landcover classification.
        Extracts features from raster data using training samples and trains selected ML models.
        """
        self.canRunInBackground = True

    def getParameterInfo(self):
        """Define tool parameters."""
        params = []
        
        # 0 - Input raster
        param = arcpy.Parameter(
            displayName = "Input Raster",
            name = "input_raster",
            datatype = "GPRasterLayer",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # 1 - Training samples
        param = arcpy.Parameter(
            displayName = "Training Samples",
            name = "training_samples",
            datatype = "GPFeatureLayer",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # 2 - Class value field
        param = arcpy.Parameter(
            displayName = "Class Value Field",
            name = "class_field",
            datatype = "Field",
            parameterType = "Required",
            direction = "Input"
        )
        param.parameterDependencies = [params[1].name]
        params.append(param)
        
        # 3 - Class name field
        param = arcpy.Parameter(
            displayName = "Class Name Field (Optional)",
            name = "class_name_field",
            datatype = "Field",
            parameterType = "Optional",
            direction = "Input"
        )
        param.parameterDependencies = [params[1].name]
        params.append(param)
        
        # 4 - Output folder
        param = arcpy.Parameter(
            displayName = "Output Folder",
            name = "output_folder",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # 5 - Models to train
        param = arcpy.Parameter(
            displayName = "Models to Train",
            name = "models_to_train",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input",
            multiValue = True
        )
        param.filter.type = "ValueList"
        param.filter.list = ["Random Forest", "XGBoost", "CatBoost"]
        param.value = ["Random Forest", "XGBoost"]
        params.append(param)
        
        # 6 - Test size
        param = arcpy.Parameter(
            displayName = "Test Set Size (%)",
            name = "test_size",
            datatype = "GPLong",
            parameterType = "Optional",
            direction = "Input"
        )
        param.value = 20
        param.filter.type = "Range"
        param.filter.list = [10, 50]
        params.append(param)
        
        # 7 - Balancing method
        param = arcpy.Parameter(
            displayName = "Class Balancing",
            name = "balancing_method",
            datatype = "GPString",
            parameterType = "Optional",
            direction = "Input"
        )
        param.filter.type = "ValueList"
        param.filter.list = ["None", "SMOTE", "Borderline SMOTE", "ADASYN", "SMOTE+Tomek", "SMOTE+ENN"]
        param.value = "SMOTE"
        params.append(param)
        
        # 8 - Imputation strategy
        param = arcpy.Parameter(
            displayName = "Missing Value Handling",
            name = "imputation_strategy",
            datatype = "GPString",
            parameterType = "Optional",
            direction = "Input"
        )
        param.filter.type = "ValueList"
        param.filter.list = ["median", "mean", "most_frequent", "zero"]
        param.value = "median"
        params.append(param)
        
        # 9 - Cross-validation folds
        param = arcpy.Parameter(
            displayName = "Cross-Validation Folds",
            name = "n_folds",
            datatype = "GPLong",
            parameterType = "Optional",
            direction = "Input"
        )
        param.value = 5
        param.filter.type = "Range"
        param.filter.list = [3, 10]
        params.append(param)
        
        # 10 - Random state
        param = arcpy.Parameter(
            displayName = "Random Seed",
            name = "random_state",
            datatype = "GPLong",
            parameterType = "Optional",
            direction = "Input"
        )
        param.value = 42
        params.append(param)
        
        # 11 - Spectral separability analysis
        param = arcpy.Parameter(
            displayName = "Assess Class Separability",
            name = "spectral_separability_analysis",
            datatype = "GPBoolean",
            parameterType = "Optional",
            direction = "Input"
        )
        param.value = True
        params.append(param)
        
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        """Execute the training tool."""
        try:
            # Initialize
            os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'
            project_dir = Path(__file__).parent
            sys.path.insert(0, str(project_dir))
            
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(" LANDCOVER CLASSIFICATION TRAINING")
            arcpy.AddMessage("=" * 70)
            
            # Parse parameters
            input_raster_param = parameters[0].valueAsText
            training_samples_param = parameters[1].valueAsText

            # Get actual file paths (ArcGIS may pass layer names instead of paths)
            try:
                desc_raster = arcpy.Describe(input_raster_param)
                input_raster = desc_raster.catalogPath
            except:
                input_raster = input_raster_param
            
            try:
                desc_samples = arcpy.Describe(training_samples_param)
                training_samples = desc_samples.catalogPath
            except:
                training_samples = training_samples_param

            arcpy.AddMessage(f"Input raster path: {input_raster}")
            arcpy.AddMessage(f"Training samples path: {training_samples}")

            class_field = parameters[2].valueAsText
            class_name_field = parameters[3].valueAsText
            output_folder = parameters[4].valueAsText
            
            # Handle multi-value parameter correctly
            if parameters[5].valueAsText:
                models_to_train = [m.strip().strip("'\"") for m in parameters[5].valueAsText.split(";")]
            else:
                models_to_train = ["Random Forest", "XGBoost"]
            
            arcpy.AddMessage(f"DEBUG: Raw parameter value: {parameters[5].valueAsText}")
            arcpy.AddMessage(f"DEBUG: Parsed models: {models_to_train}")
            
            test_size = parameters[6].value / 100.0 if parameters[6].value else 0.2
            balancing_method = parameters[7].valueAsText
            imputation_strategy = parameters[8].valueAsText
            n_folds = parameters[9].value if parameters[9].value else 5
            random_state = parameters[10].value if parameters[10].value else 42
            do_spectral_analysis = parameters[11].value
            
            # Map parameter names
            balancing_map = {
                "None": "none",
                "SMOTE": "smote",
                "Borderline SMOTE": "bsmote",
                "ADASYN": "adasyn",
                "SMOTE+Tomek": "smote_tomek",
                "SMOTE+ENN": "smote_enn"
            }
            balancing_technique = balancing_map.get(balancing_method, "smote")
            
            arcpy.AddMessage("\nConfiguration:")
            arcpy.AddMessage(f"Models: {', '.join(models_to_train)}")
            arcpy.AddMessage(f"Test size: {test_size*100:.0f}%")
            arcpy.AddMessage(f"Balancing: {balancing_method}")
            arcpy.AddMessage(f"Imputation: {imputation_strategy}")
            arcpy.AddMessage(f"CV folds: {n_folds}")
            arcpy.AddMessage(f"Random seed: {random_state}")
            
            # Setup directories
            output_path = Path(output_folder)
            data_dir = output_path / "data"
            plots_dir = data_dir / "plots"
            models_dir = data_dir / "model_files"
            csv_dir = data_dir / "csv"
            raster_dir = data_dir / "raster"
            
            for directory in [data_dir, plots_dir, models_dir, csv_dir, raster_dir]:
                directory.mkdir(parents = True, exist_ok = True)
            
            
            # Create temporary files
            temp_dir = Path(arcpy.env.scratchFolder)
            temp_samples = temp_dir / f"training_samples_{dt.now().strftime('%Y%m%d_%H%M%S')}.shp"
            temp_raster = temp_dir / f"input_raster_{dt.now().strftime('%Y%m%d_%H%M%S')}.tif"
                        
            # Import config
            spec = importlib.util.spec_from_file_location("config", project_dir / "config.py")
            config_module = importlib.util.module_from_spec(spec)
            sys.modules["config"] = config_module
            spec.loader.exec_module(config_module)
            
            # Update config
            config_module.RASTER_PATH = str(temp_raster)
            config_module.SHP_PATH = str(temp_samples)
            config_module.TARGET_COLUMN = class_field
            config_module.TARGET_COLUMN_DESC = class_name_field if class_name_field else class_field
            config_module.TEST_SIZE = test_size
            config_module.RANDOM_STATE = random_state
            config_module.N_SPLITS = n_folds
            config_module.IMPUTATION_STRATEGY = imputation_strategy
            config_module.BALANCING_TECHNIQUE = balancing_technique
            config_module.DATA_DIR = str(data_dir)
            config_module.PLOTS_PATH = str(plots_dir)
            config_module.MODELS_PATH = str(models_dir)
            config_module.CSV_PATH = str(csv_dir)
            config_module.CLASSIFICATION_PATH = str(raster_dir)
            config_module.DIR_LIST = [str(plots_dir), str(models_dir), str(csv_dir), str(raster_dir)]

            # Force reload of modules that might have imported config already
            if 'evaluation' in sys.modules:
                importlib.reload(sys.modules['evaluation'])
            if 'evaluation.visualizer' in sys.modules:
                importlib.reload(sys.modules['evaluation.visualizer'])

            arcpy.AddMessage(f"Plots path set to: {plots_dir}")
            arcpy.AddMessage(f"CSV path set to: {csv_dir}")
            arcpy.AddMessage(f"Models path set to: {models_dir}")
            
            # arcpy.AddMessage("\n[3/6] Loading modules...")

            from utils import DataProcessor
            from utils import JeffriesMatusitaDistance    
            from models import train_ml_model, RFClassificationModel, XGBClassificationModel, CatBoostClassificationModel

            arcpy.AddMessage("\n[1/6] Preparing data...")
            arcpy.SetProgressor("step", "Preparing data...", 0, 100, 5)
            
            arcpy.FeatureClassToFeatureClass_conversion(
                training_samples,
                str(temp_samples.parent),
                temp_samples.name
            )

            # Verify file exists
            if not Path(input_raster).exists():
                raise FileNotFoundError(f"Raster file not found: {input_raster}")
            
            with rio.open(input_raster) as src:
                band_count = src.count
                # Get band descriptions - this is what we need to preserve!
                band_names = []
                for i in range(band_count):
                    desc = src.descriptions[i]
                    if desc and desc.strip():
                        band_names.append(desc.strip())
                    else:
                        band_names.append(f'Band_{i+1}')
                
                arcpy.AddMessage(f"Found {len(band_names)} bands:")
                for bn in band_names:
                    arcpy.AddMessage(f"    - {bn}")
                    
            arcpy.CopyRaster_management(input_raster, str(temp_raster))
            
            arcpy.SetProgressorPosition(5)
            
            # Import and configure modules
            arcpy.AddMessage("\n[2/6] Loading modules...")

            
            # Get class information
            arcpy.AddMessage("\n[3/6] Analysing class distribution...")
            arcpy.SetProgressorPosition(10)
            
            fields = [class_field]
            if class_name_field:
                fields.append(class_name_field)
            
            class_info = {}
            class_counts = {}
            
            with arcpy.da.SearchCursor(training_samples, fields) as cursor:
                for row in cursor:
                    class_val = int(row[0])
                    class_name = row[1] if class_name_field else f"Class_{class_val}"
                    if class_val not in class_info:
                        class_info[class_val] = class_name
                        class_counts[class_val] = 0
                    class_counts[class_val] += 1
            
            config_module.CLASS_NAMES = class_info
            config_module.NUM_CLASSES = len(class_info)
            
            arcpy.AddMessage(f"\nDetected {len(class_info)} classes:")
            for class_val, class_name in sorted(class_info.items()):
                arcpy.AddMessage(f"  {class_val}: {class_name} ({class_counts[class_val]} features)")
            
            # Generate LULC specifications
            lulc_specs = {class_val: 100 for class_val in class_info.keys()}
            config_module.LULC_SPECIFICATIONS = lulc_specs
           
            # Initialize DataProcessor
            data_processor = DataProcessor(str(temp_raster), str(temp_samples))
            
            # Extract features
            arcpy.AddMessage("\n[4/6] Extracting features from raster...")
            arcpy.SetProgressorLabel("Extracting features...")
            arcpy.SetProgressorPosition(15)
            
            desc = arcpy.Describe(training_samples)
            arcpy.AddMessage(f"  Geometry type: {desc.shapeType}")
            
            # Extract raster values based on geometry type
            if desc.shapeType == "Polygon":
                arcpy.AddMessage("  Extracting polygon values...")
                raster_gdf = data_processor.extract_polygon_values(
                    # lulc_specs = lulc_specs,
                    class_field = class_field,
                    class_name_field = class_name_field if class_name_field else None
                )
            else:
                arcpy.AddMessage("  Extracting point values...")
                points_gdf = data_processor.extract_points_by_lulc(lulc_specs)
                raster_gdf = data_processor.extract_raster_values(points_gdf)
            
            arcpy.AddMessage(f"  Extracted {len(raster_gdf)} samples")
            arcpy.SetProgressorPosition(25)
            
            # Preprocess data
            arcpy.AddMessage("\n  Preprocessing data...")
            
            fields_to_remove = []
            if class_name_field and class_name_field in raster_gdf.columns:
                fields_to_remove.append(class_name_field)
            
            raster_gdf = raster_gdf[[col for col in raster_gdf.columns.unique() 
                                    if col not in fields_to_remove]]
            
            # Filter problematic pixels
            cols = [col for col in raster_gdf.columns.unique() if col != class_field]
            raster_gdf[class_field] = pd.to_numeric(raster_gdf[class_field], errors = 'coerce')
            raster_gdf_mask = raster_gdf[~(raster_gdf[cols].abs() < config_module.TOLERANCE).any(axis = 1)]
            
            feature_columns = [col for col in raster_gdf_mask.columns.unique() 
                             if col != class_field]
            
            arcpy.AddMessage(f"Features: {len(feature_columns)} bands")
            arcpy.AddMessage(f"Samples after filtering: {len(raster_gdf_mask)}")
            

            # If feature columns don't have proper names, use band names
            if len(feature_columns) == len(band_names):
                arcpy.AddMessage("Mapping columns to band names...")
                # Create mapping from current columns to band names
                column_mapping = dict(zip(feature_columns, band_names))
                raster_gdf_mask = raster_gdf_mask.rename(columns = column_mapping)
                feature_columns = band_names
            
            arcpy.AddMessage(f"Features: {feature_columns}")
            arcpy.AddMessage(f"Samples after filtering: {len(raster_gdf_mask)}")
            
            # Spectral separability analysis
            if do_spectral_analysis:
                arcpy.AddMessage("\n[5/6] Performing spectral separability analysis...")
                arcpy.SetProgressorLabel("Spectral analysis...")
                arcpy.SetProgressorPosition(30)
                
                try:
                   
                    
                    jm_df = raster_gdf_mask.copy()
                    analyser = JeffriesMatusitaDistance(
                        df = jm_df,
                        spectral_columns = feature_columns,
                        class_names = class_info,
                        class_column = class_field
                    )
                    
                    # jm_matrix = analyser.compute_pairwise_distances()
                    
                    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
                    jm_path = csv_dir / "jm_distance_matrix.csv"
                    analyser.save_matrix(str(jm_path))
                    
                    heatmap_path = plots_dir / "jm_distance_heatmap.png"
                    analyser.plot_heatmap(
                        figsize = (12, 10),
                        cmap = 'YlOrRd',
                        annot = True,
                        save_path = str(heatmap_path)
                    )
                    
                    summary = analyser.get_separability_summary()
                    
                    arcpy.AddMessage(f"Excellent separability: {len(summary['excellent'])} pairs")
                    arcpy.AddMessage(f"Moderate separability: {len(summary['moderate'])} pairs")
                    arcpy.AddMessage(f"Poor separability: {len(summary['poor'])} pairs")
                    
                    summary_path = csv_dir / "spectral_separability_summary.txt"
                    with open(summary_path, 'w') as f:
                        f.write("Spectral Separability Analysis Summary\n")
                        f.write("=" * 60 + "\n\n")
                        
                        f.write("Excellent Separability (JM > 1.9):\n")
                        for pair, jm_value in summary['excellent']:
                            f.write(f"  {pair}: {jm_value:.3f}\n")
                        f.write(f"  Total:{len(summary['excellent'])} pairs\n\n")
                        
                        f.write("Moderate Separability (1.0 < JM <= 1.9):\n")
                        for pair, jm_value in summary['moderate']:
                            f.write(f"  {pair}: {jm_value:.3f}\n")
                        f.write(f"  Total:{len(summary['moderate'])} pairs\n\n")
                        
                        f.write("Poor Separability (JM <= 1.0):\n")
                        for pair, jm_value in summary['poor']:
                            f.write(f"{pair}: {jm_value:.3f}\n")
                        f.write(f"  Total: {len(summary['poor'])} pairs\n\n")
                    
                    arcpy.AddMessage(f"Saved to: {jm_path.name}")
                    
                except Exception as e:
                    arcpy.AddWarning(f"Spectral analysis failed: {str(e)}")
            else:
                arcpy.AddMessage("\n[5/6] Skipping spectral separability analysis...")
            
            arcpy.SetProgressorPosition(35)
            
            # Prepare training data
            arcpy.AddMessage("\n[6/6] Training models...")
            arcpy.SetProgressorLabel("Preparing training data...")

            gdf, _, _, X, y = data_processor.get_features_and_targets(
                feature_columns = feature_columns,
                target_column = class_field,
                geodataframe = raster_gdf_mask,
                imputation_strategy = imputation_strategy
            )
            
            arcpy.AddMessage(f"Feature matrix: {X.shape}")
            arcpy.AddMessage(f"Target vector: {y.shape}")

            
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size = test_size,
                random_state = random_state,
                stratify = y
            )
            
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            arcpy.AddMessage(f"Training samples: {len(X_train)}")
            arcpy.AddMessage(f"Test samples: {len(X_test)}")
            
            # Initialize models
            ml_models = []
            for model_name in models_to_train:
                if model_name == "Random Forest":
                    ml_models.append(RFClassificationModel(n_splits = n_folds))
                elif model_name == "XGBoost":
                    ml_models.append(XGBClassificationModel(n_splits = n_folds))
                elif model_name == "CatBoost":
                    ml_models.append(CatBoostClassificationModel(n_splits = n_folds))
                else:
                    arcpy.AddWarning(f"Unknown model: {model_name} - skipping")
            
            if len(ml_models) == 0:
                raise ValueError("No valid models selected. Please select at least one model to train.")
            
            arcpy.AddMessage(f"\nModels to train: {[m.__class__.__name__ for m in ml_models]}")


            # Train each model
            all_results = {}
            base_progress = 40
            progress_per_model = 50 / len(ml_models) 
            model_times = {} 
            
            for idx, model in enumerate(ml_models):
                model_name = model.__class__.__name__
                
                arcpy.AddMessage(f"\nTraining {model_name}...")
                # Include time to track how long the training takes  
                arcpy.AddMessage(f"  {'-' * 50}")
                arcpy.SetProgressorLabel(f"Training {model_name}...")
                arcpy.SetProgressorPosition(int(base_progress + idx * progress_per_model))

                # Track time spent training the model 
               
                # Prepare balancing parameters
                balancing_params = {
                    'k_neighbors': 5,
                    'm_neighbors': 10,
                    'random_state': random_state
                }

                model_start_time = time.time()
                
                if balancing_technique != 'none' and len(X_train) > 20000:
                    arcpy.AddWarning(f"Large dataset ({len(X_train)} samples). Disabling balancing for safety.")
                    balancing_technique = 'none'  # Disable balancing for large datasets
                    arcpy.AddWarning("Continuing without balancing to prevent memory issues.")

                if balancing_technique == 'smote':
                    relevant_params = {k: v for k, v in balancing_params.items() 
                                     if k in ['k_neighbors', 'random_state']}
                elif balancing_technique == 'bsmote':
                    relevant_params = {k: v for k, v in balancing_params.items() 
                                     if k in ['k_neighbors', 'm_neighbors', 'random_state']}
                elif balancing_technique in ['adasyn', 'smote_tomek', 'smote_enn']:
                    relevant_params = {k: v for k, v in balancing_params.items() 
                                     if k in ['k_neighbors', 'random_state']}
                else:
                    relevant_params = {}
                
                try:
                    results = train_ml_model(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        feature_columns,
                        balancing_technique = balancing_technique,
                        imputation_strategy = imputation_strategy,
                        **relevant_params
                    )

                    # Record model training time 
                    model_elapsed = time.time() - model_start_time
                    model_times[model_name] = model_elapsed
                    
                    results['balancing_technique'] = balancing_technique
                    results['imputation_strategy'] = imputation_strategy
                    all_results[model_name] = results
                    
                    arcpy.AddMessage("Training completed!")
                    arcpy.AddMessage(f"Test Accuracy: {results['test_accuracy']:.4f}")
                    
                    if 'cv_results' in results:
                        cv_acc = results['cv_results']['avg_accuracy']
                        cv_std = results['cv_results']['std_accuracy']
                        arcpy.AddMessage(f"CV Accuracy: {cv_acc:.4f} (±{cv_std:.4f})")
                    
                except Exception as e:
                    model_elapsed = time.time() - model_start_time
                    model_times[model_name] = model_elapsed
                    arcpy.AddWarning(f"Failed to train {model_name} after {timedelta(seconds = int(model_elapsed))}")
                    arcpy.AddWarning(f"Failed to train {model_name}")
                    arcpy.AddWarning(f"Error: {str(e)}")
                    arcpy.AddWarning(traceback.format_exc())
            
            # Calculate total training time
            total_time = sum(model_times.values())
            arcpy.AddMessage("\nModel training times:")
            for model_name, t in model_times.items():
                arcpy.AddMessage(f"{model_name}: {timedelta(seconds = int(t))}")
            arcpy.AddMessage(f"Total training time: {timedelta(seconds = int(total_time))}")
            if len(model_times) > 0: 
                arcpy.AddMessage(f"Average time per model: {timedelta(seconds = int(total_time / len(model_times)))}")
            
            # Continue to monitor progress 
            arcpy.SetProgressorPosition(90)
            
            # Save summary
            arcpy.AddMessage("\nSaving results...")
            arcpy.SetProgressorLabel("Saving results...")
            
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            summary_data = []
            
            for model_name, results in all_results.items():
                row = {
                    'Model': model_name,
                    'Balancing_Technique': results.get('balancing_technique', 'none'),
                    'Imputation_Strategy': results.get('imputation_strategy', 'none'),
                    'Test_Accuracy': results['test_accuracy']
                }
                
                if 'cv_results' in results:
                    row.update({
                        'CV_Accuracy': results['cv_results']['avg_accuracy'],
                        'CV_Std': results['cv_results']['std_accuracy'],
                        'Avg_Recall': results['cv_results']['recall'],
                        'Avg_Precision': results['cv_results']['precision'],
                        'Avg_F1': results['cv_results']['f1_score']
                    })
                
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = csv_dir / "model_comparison.csv"
            summary_df.to_csv(summary_path, index = False)
            
            # Save configuration
            config_dict = {
                'input_raster': input_raster,
                'training_samples': training_samples,
                'class_field': class_field,
                'class_name_field': class_name_field,
                'test_size': test_size,
                'balancing_method': balancing_method,
                'imputation_strategy': imputation_strategy,
                'n_folds': n_folds,
                'random_state': random_state,
                'class_names': class_info,
                'band_names': band_names,
                'feature_columns': feature_columns,
                'num_classes': len(class_info),
                'num_bands': len(band_names),
                'timestamp': timestamp
            }
            
            config_path = output_path / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent = 2)
            
            # Save class names 
            class_names_path = output_path / "class_names.json"
            with open(class_names_path, 'w') as f:
                json.dump({
                    'class_names': class_info,
                    'class_field': class_field,
                    'class_name_field': class_name_field
                }, f, indent = 2)

            # Save band names
            band_names_path = output_path / "band_names.json"
            with open(band_names_path, 'w') as f:
                json.dump({
                    'band_names': band_names,
                    'feature_columns': feature_columns,
                    'num_bands': len(band_names)
                }, f, indent = 2)

            arcpy.SetProgressorPosition(100)
            
            # Cleanup
            arcpy.AddMessage("\nCleaning up temporary files...")
            for temp_file in [temp_samples, temp_raster]:
                if temp_file.exists():
                    try:
                        for ext_file in temp_file.parent.glob(f"{temp_file.stem}.*"):
                            ext_file.unlink(missing_ok = True)
                    except:
                        pass
            
            # Final summary
            arcpy.AddMessage("\n" + "=" * 70)
            arcpy.AddMessage(" TRAINING COMPLETED SUCCESSFULLY!")
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(f"\nOutput Location: {output_folder}")
            arcpy.AddMessage(f"  Models: {models_dir}")
            arcpy.AddMessage(f"  Reports: {csv_dir}")
            arcpy.AddMessage(f"  Plots: {plots_dir}")
            
            arcpy.AddMessage("\nModel Performance Summary:")
            arcpy.AddMessage("-" * 50)
            for model_name, results in all_results.items():
                acc = results.get('test_accuracy', 0)
                arcpy.AddMessage(f"  {model_name}: {acc:.4f}")
            
        except Exception as e:
            arcpy.AddError("\nERROR: Training failed")
            arcpy.AddError(f"  {str(e)}")
            arcpy.AddError("\nTraceback:")
            arcpy.AddError(traceback.format_exc())
            raise


class PredictLandcover(object):
    """Apply a trained landcover model to generate a classification map."""
    
    def __init__(self):
        self.label = "2. Predict Landcover"
        self.description = "Apply trained model to create landcover classification map"
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []
        
        # Model file
        param = arcpy.Parameter(
            displayName = "Trained Model File",
            name = "model_file",
            datatype = "DEFile",
            parameterType = "Required",
            direction = "Input"
        )
        param.filter.list = ["joblib"]
        params.append(param)
        
        # Scaler file
        param = arcpy.Parameter(
            displayName = "Model Scaler File",
            name = "scaler_file",
            datatype = "DEFile",
            parameterType = "Required",
            direction = "Input"
        )
        param.filter.list = ["joblib"]
        params.append(param)
        
        # Input raster
        param = arcpy.Parameter(
            displayName = "Input Raster for Prediction",
            name = "input_raster",
            datatype = "GPRasterLayer",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # Output raster
        param = arcpy.Parameter(
            displayName = "Output Classification Raster",
            name = "output_raster",
            datatype = "DERasterDataset",
            parameterType = "Required",
            direction = "Output"
        )
        params.append(param)
        
        # Model type
        param = arcpy.Parameter(
            displayName = "Model Type",
            name = "model_type",
            datatype = "GPString",
            parameterType = "Required",
            direction = "Input"
        )
        param.filter.type = "ValueList"
        param.filter.list = ["Random Forest", "XGBoost", "CatBoost"]
        param.value = "XGBoost"
        params.append(param)
        
        return params

    def execute(self, parameters, messages):
        try:
            os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'
            project_dir = Path(__file__).parent
            sys.path.insert(0, str(project_dir))
            
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(" LANDCOVER PREDICTION")
            arcpy.AddMessage("=" * 70)
            
            model_file = parameters[0].valueAsText
            scaler_file = parameters[1].valueAsText
            input_raster = parameters[2].valueAsText
            output_raster = parameters[3].valueAsText
            model_type = parameters[4].valueAsText
            
            arcpy.AddMessage(f"\nModel: {model_type}")
            arcpy.AddMessage(f"Input: {Path(input_raster).name}")
            arcpy.AddMessage(f"Output: {Path(output_raster).name}")
            
            # Create temporary raster
            temp_dir = Path(arcpy.env.scratchFolder)
            temp_raster = temp_dir / f"prediction_input_{dt.now().strftime('%Y%m%d_%H%M%S')}.tif"
            
            arcpy.AddMessage("\nPreparing data...")
            arcpy.SetProgressor("step", "Preparing data...", 0, 100, 10)
            arcpy.CopyRaster_management(input_raster, str(temp_raster))
            arcpy.SetProgressorPosition(20)
            
            # Import classifier
            arcpy.AddMessage(f"\nLoading {model_type} classifier...")
            
            if model_type == "Random Forest":
                from classification import RFLandcoverClassifier
                classifier = RFLandcoverClassifier(model_file, scaler_file, num_classes = 12, class_offset = 0)
            elif model_type == "XGBoost":
                from classification import XGBoostLandcoverClassifier
                classifier = XGBoostLandcoverClassifier(model_file, scaler_file, num_classes = 12)
            elif model_type == "CatBoost":
                from classification import CatBoostLandcoverClassifier
                classifier = CatBoostLandcoverClassifier(model_file, scaler_file, num_classes = 12)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Run prediction
            arcpy.AddMessage("\nClassifying raster...")
            arcpy.SetProgressorLabel("Classifying...")
            arcpy.SetProgressorPosition(40)
            
            classifier.predict(str(temp_raster), output_raster)
            
            arcpy.SetProgressorPosition(100)
            
            # Cleanup
            if temp_raster.exists():
                for ext_file in temp_raster.parent.glob(f"{temp_raster.stem}.*"):
                    ext_file.unlink(missing_ok = True)
            
            arcpy.AddMessage("\n" + "=" * 70)
            arcpy.AddMessage(" PREDICTION COMPLETED!")
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(f"\nOutput: {output_raster}")
            
        except Exception as e:
            arcpy.AddError("\nERROR: Prediction failed")
            arcpy.AddError(f"{str(e)}")
            arcpy.AddError(traceback.format_exc())
            raise


class SpectralSeparabilityAnalysis(object):
    """Perform spectral separability analysis on training data."""
    
    def __init__(self):
        self.label = "3. Spectral Separability Analysis"
        self.description = "Perform Jeffries-Matusita spectral separability analysis"
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []
        
        # Input raster
        param = arcpy.Parameter(
            displayName = "Input Raster",
            name = "input_raster",
            datatype = "GPRasterLayer",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # Training samples
        param = arcpy.Parameter(
            displayName = "Training Samples",
            name = "training_samples",
            datatype = "GPFeatureLayer",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # Class field
        param = arcpy.Parameter(
            displayName = "Class Value Field",
            name = "class_field",
            datatype = "Field",
            parameterType = "Required",
            direction = "Input"
        )
        param.parameterDependencies = [params[1].name]
        params.append(param)
        
        # Class name field
        param = arcpy.Parameter(
            displayName = "Class Name Field (Optional)",
            name = "class_name_field",
            datatype = "Field",
            parameterType = "Optional",
            direction = "Input"
        )
        param.parameterDependencies = [params[1].name]
        params.append(param)
        
        # Output folder
        param = arcpy.Parameter(
            displayName = "Output Folder",
            name = "output_folder",
            datatype = "DEFolder",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # Samples per class
        param = arcpy.Parameter(
            displayName = "Samples per Class",
            name = "samples_per_class",
            datatype = "GPLong",
            parameterType = "Optional",
            direction = "Input"
        )
        param.value = 100
        param.filter.type = "Range"
        param.filter.list = [10, 1000]
        params.append(param)
        
        return params

    def execute(self, parameters, messages):
        try:
            os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'
            project_dir = Path(__file__).parent
            sys.path.insert(0, str(project_dir))
            
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(" SPECTRAL SEPARABILITY ANALYSIS")
            arcpy.AddMessage("=" * 70)
            
            input_raster = parameters[0].valueAsText
            training_samples = parameters[1].valueAsText
            class_field = parameters[2].valueAsText
            class_name_field = parameters[3].valueAsText
            output_folder = parameters[4].valueAsText
            samples_per_class = parameters[5].value if parameters[5].value else 100
            
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok = True)
            
            # Temporary files
            temp_dir = Path(arcpy.env.scratchFolder)
            # timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            temp_samples = temp_dir / "spectral_samples.shp"
            temp_raster = temp_dir / "spectral_raster.tif"
            
            arcpy.AddMessage("\nPreparing data...")
            arcpy.SetProgressor("step", "Preparing data...", 0, 100, 10)
            
            arcpy.FeatureClassToFeatureClass_conversion(
                training_samples, str(temp_samples.parent), temp_samples.name
            )
            arcpy.CopyRaster_management(input_raster, str(temp_raster))
            arcpy.SetProgressorPosition(20)
            
            # Get class info
            fields = [class_field]
            if class_name_field:
                fields.append(class_name_field)
            
            class_info = {}
            with arcpy.da.SearchCursor(training_samples, fields) as cursor:
                for row in cursor:
                    class_val = int(row[0])
                    class_name = row[1] if class_name_field else f"Class_{class_val}"
                    if class_val not in class_info:
                        class_info[class_val] = class_name
            
            lulc_specs = {class_val: samples_per_class for class_val in class_info.keys()}
            
            arcpy.AddMessage(f"\nAnalyzing {len(class_info)} classes:")
            for class_val, class_name in sorted(class_info.items()):
                arcpy.AddMessage(f"  {class_val}: {class_name}")
            
            # Extract data
            arcpy.AddMessage("\nExtracting spectral data...")
            arcpy.SetProgressorPosition(30)
            
            
            data_processor = DataProcessor(str(temp_raster), str(temp_samples))
            
            desc = arcpy.Describe(training_samples)
            if desc.shapeType == "Polygon":
                raster_gdf = data_processor.extract_polygon_values(
                    lulc_specs = lulc_specs,
                    class_field = class_field,
                    class_name_field = class_name_field if class_name_field else None
                )
            else:
                points_gdf = data_processor.extract_points_by_lulc(lulc_specs)
                raster_gdf = data_processor.extract_raster_values(points_gdf)
            
            arcpy.SetProgressorPosition(50)
            
            # Prepare features
            fields_to_remove = ["geometry", class_field]
            if class_name_field:
                fields_to_remove.append(class_name_field)
            
            feature_columns = [col for col in raster_gdf.columns.unique() 
                             if col not in fields_to_remove]
            
            cols = feature_columns + [class_field]
            analysis_df = raster_gdf[cols].copy()
            analysis_df = analysis_df.dropna()
            
            arcpy.AddMessage(f"Samples: {len(analysis_df)}")
            arcpy.AddMessage(f"Bands: {len(feature_columns)}")
            
            # Perform JM analysis
            arcpy.AddMessage("\nComputing Jeffries-Matusita distances...")
            arcpy.SetProgressorPosition(60)
            
            
            
            analyser = JeffriesMatusitaDistance(
                df = analysis_df,
                spectral_columns = feature_columns,
                class_names = class_info,
                class_column = class_field
            )
            
            jm_matrix = analyser.compute_pairwise_distances()
            arcpy.SetProgressorPosition(70)
            
            # Save results
            jm_csv = output_path / "jm_distance_matrix.csv"
            analyser.save_matrix(str(jm_csv))
            
            heatmap_png = output_path / "jm_heatmap.png"
            analyser.plot_heatmap(
                figsize = (12, 10),
                cmap = 'YlOrRd',
                annot = True,
                fmt = '.3f',
                save_path = str(heatmap_png)
            )
            
            summary = analyser.get_separability_summary()
            
            # Save text summary
            summary_txt = output_path / "separability_summary.txt"
            with open(summary_txt, 'w') as f:
                f.write("SPECTRAL SEPARABILITY ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Date: {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Samples: {len(analysis_df)}\n")
                f.write(f"Bands: {len(feature_columns)}\n")
                f.write(f"Classes: {len(class_info)}\n\n")
                
                f.write("Excellent Separability (JM > 1.9):\n")
                for pair, jm_val in summary['excellent']:
                    f.write(f"  {pair}: {jm_val:.3f}\n")
                f.write(f"Total: {len(summary['excellent'])}\n\n")
                
                f.write("Moderate Separability (1.0 < JM <= 1.9):\n")
                for pair, jm_val in summary['moderate']:
                    f.write(f"  {pair}: {jm_val:.3f}\n")
                f.write(f"Total: {len(summary['moderate'])}\n\n")
                
                f.write("Poor Separability (JM <= 1.0):\n")
                for pair, jm_val in summary['poor']:
                    f.write(f"  {pair}: {jm_val:.3f}\n")
                f.write(f"Total: {len(summary['poor'])}\n")
            
            arcpy.SetProgressorPosition(100)
            
            # Cleanup
            for temp_file in [temp_samples, temp_raster]:
                if temp_file.exists():
                    try:
                        for ext_file in temp_file.parent.glob(f"{temp_file.stem}.*"):
                            ext_file.unlink(missing_ok = True)
                    except:
                        pass
            
            arcpy.AddMessage("\n" + "=" * 70)
            arcpy.AddMessage(" ANALYSIS COMPLETED!")
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(f"\nResults: {output_folder}")
            arcpy.AddMessage(f"\nSummary:")
            arcpy.AddMessage(f"  Excellent: {len(summary['excellent'])} pairs")
            arcpy.AddMessage(f"  Moderate: {len(summary['moderate'])} pairs")
            arcpy.AddMessage(f"  Poor: {len(summary['poor'])} pairs")
            
        except Exception as e:
            arcpy.AddError(f"\nERROR: Analysis failed")
            arcpy.AddError(f"  {str(e)}")
            arcpy.AddError(traceback.format_exc())
            raise


class ApplyMajorityFilter(object):
    """Apply majority filter to classification raster."""
    
    def __init__(self):
        self.label = "4. Apply Majority Filter"
        self.description = "Apply majority filter to smooth classification results"
        self.canRunInBackground = True

    def getParameterInfo(self):
        params = []
        
        # Input raster
        param = arcpy.Parameter(
            displayName = "Input Classified Raster",
            name = "input_raster",
            datatype = "GPRasterLayer",
            parameterType = "Required",
            direction = "Input"
        )
        params.append(param)
        
        # Output raster
        param = arcpy.Parameter(
            displayName = "Output Filtered Raster",
            name = "output_raster",
            datatype = "DERasterDataset",
            parameterType = "Required",
            direction = "Output"
        )
        params.append(param)
        
        # Window size
        param = arcpy.Parameter(
            displayName = "Window Size",
            name = "window_size",
            datatype = "GPLong",
            parameterType = "Optional",
            direction = "Input"
        )
        param.value = 3
        param.filter.type = "ValueList"
        param.filter.list = [3, 5, 7, 9]
        params.append(param)
        
        return params

    def execute(self, parameters, messages):
        try:
            os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'
            project_dir = Path(__file__).parent
            sys.path.insert(0, str(project_dir))
            
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(" APPLY MAJORITY FILTER")
            arcpy.AddMessage("=" * 70)
            
            input_raster = parameters[0].valueAsText
            output_raster = parameters[1].valueAsText
            window_size = parameters[2].value if parameters[2].value else 3
            
            arcpy.AddMessage(f"\nWindow size: {window_size}x{window_size}")
            
            # Temporary files
            temp_dir = Path(arcpy.env.scratchFolder)
            timestamp = dt.now().strftime('%Y%m%d_%H%M%S')
            temp_input = temp_dir / "filter_input.tif"
            temp_output = temp_dir / "filter_output.tif"
            
            arcpy.AddMessage("\nPreparing data...")
            arcpy.SetProgressor("step", "Preparing...", 0, 100, 20)
            arcpy.CopyRaster_management(input_raster, str(temp_input))
            arcpy.SetProgressorPosition(30)
            
            # Apply filter
            arcpy.AddMessage("\nApplying majority filter...")
            arcpy.SetProgressorLabel("Filtering...")
            
            from majority_filter import majority_filter
            
            majority_filter(str(temp_input), str(temp_output), window_size = window_size)
            arcpy.SetProgressorPosition(70)
            
            # Copy to output
            arcpy.AddMessage("\nSaving result...")
            arcpy.CopyRaster_management(str(temp_output), output_raster)
            arcpy.SetProgressorPosition(100)
            
            # Cleanup
            for temp_file in [temp_input, temp_output]:
                if temp_file.exists():
                    for ext_file in temp_file.parent.glob(f"{temp_file.stem}.*"):
                        ext_file.unlink(missing_ok = True)
            
            arcpy.AddMessage("\n" + "=" * 70)
            arcpy.AddMessage(" FILTER APPLIED SUCCESSFULLY!")
            arcpy.AddMessage("=" * 70)
            arcpy.AddMessage(f"\nOutput: {output_raster}")
            
        except Exception as e:
            arcpy.AddError(f"\nERROR: Filter failed")
            arcpy.AddError(f"  {str(e)}")
            arcpy.AddError(traceback.format_exc())
            raise