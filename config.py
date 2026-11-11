from pathlib import Path 

# Paths 
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
DIR_LIST  = [DATA_DIR / 'plots', 
             DATA_DIR / 'model_files', 
             DATA_DIR / 'csv', 
             DATA_DIR / 'raster',
             ]

PLOTS_PATH = DIR_LIST[0]
MODELS_PATH = DIR_LIST[1]
CSV_PATH = DIR_LIST[2]
CLASSIFICATION_PATH = DIR_LIST[3]



# Model specs 
YEAR = ''
LOC = "GAW"
NUM_CLASSES = 12
NUM_BANDS = 8
N_SPLITS = 5


# Data paths
RASTER_PATH = 'GAW_S2_Demo.tif'
SHP_PATH = 'demo_lulc_categories.geojson'

# Visualization settings
DEFAULT_PLOT_STYLE = 'ggplot'
EXPORT_RESOLUTION = 500
FIGURE_SIZE = (10, 6)
FIELDS_TO_REMOVE = ["Class"] #, "geometry"

CLASS_NAMES = {
                1: 'Water',
                2: 'Mangrove',
                3: 'Swamp',
                4: 'Forest',
                5: 'Grassland',
                6: 'Cropland',
                7: 'Built-up',
                8: 'Bareland',
                9: 'Coconut',
                10: 'Oil Palm',
                11: 'Rubber',
                12: 'Mining'
              }

LULC_SPECIFICATIONS = {
                1: 100, 
                2: 100,
                3: 350,
                4: 680,
                5: 100,
                6: 270,
                7: 100,
                8: 100,
                9: 756,
                10: 100,
                11: 100,
                12: 100,
            }

TARGET_COLUMN = 'Value'
TARGET_COLUMN_DESC = 'Class'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Value for masking problematic pixels
TOLERANCE = 1e-8

# Imputation strategies
IMPUTATION_STRATEGY = 'median' # 'mean', 'median', 'most_frequent', 'zero'

# Balancing configurations with imputation
BALANCING_TECHNIQUE = 'smote' # 'none', 'smote', 'adasyn', 'bsmote', 'smote_tomek', 'smote_enn'

BALANCING_PARAMS = {
    'k_neighbors': 5,  # For SMOTE and BorderlineSMOTE
    'm_neighbors': 10,  # For BorderlineSMOTE only
    'random_state': 42
}