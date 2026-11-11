import config
from pathlib import Path
from config import (CLASSIFICATION_PATH, MODELS_PATH, RASTER_PATH)
from classification import RFLandcoverClassifier, XGBoostLandcoverClassifier, CatBoostLandcoverClassifier

models_path = Path(MODELS_PATH)
raster_path = Path(CLASSIFICATION_PATH)

# Make predictions using XGBoost Model
image_predictor = XGBoostLandcoverClassifier(
    model_path = models_path / 'XGBClassificationModel.joblib', # type: ignore
    scaler_path = models_path / 'XGBClassificationModel_scaler.joblib',  # type: ignore
    num_classes = config.NUM_CLASSES
)

image_predictor.predict(
    input_raster = RASTER_PATH,
    output_path = raster_path / f'LANDCOVER_XGB_{config.YEAR}.tif' # type: ignore
)   