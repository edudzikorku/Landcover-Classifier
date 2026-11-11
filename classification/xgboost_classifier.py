import os
import joblib # type: ignore
import numpy as np # type: ignore
import rasterio as rio # type: ignore
from typing import Tuple # type: ignore

class XGBoostLandcoverClassifier:
    """
    Predictor for XGBoost models. Utilises trained model and scaler to run 
    landcover classification on raster data.
    """

    def __init__(self, model_path: str, scaler_path: str, num_classes: int = 12) -> None:
        """
        Initialize XGBoost predictor

        Args:
            model_path (str): Path to XGBoost model (.joblib)
            scaler_path (str): Path to corresponding scaler
            num_classes (int): Number of landcover classes (default: 12)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.num_classes = num_classes
        self.model = None
        self.scaler = None
        self.nan_mask = None
        self._load_assets()

    def _load_assets(self) -> None:
        """Load both model and scaler"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/scaler: {str(e)}")

    def _preprocess_input(self, raster: np.ndarray) -> np.ndarray:
        """Prepare raster data for prediction"""
        # Reshape to (height * width, bands)
        reshaped = raster.transpose(1, 2, 0).reshape(-1, raster.shape[0])

        # Create mask for valid (non-NaN) pixels
        self.nan_mask = ~np.isnan(reshaped).any(axis = 1)
        valid_data = reshaped[self.nan_mask]

        # Apply scaling
        return self.scaler.transform(valid_data) # type: ignore

    def _postprocess_output(self, predictions: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """Convert 0-based predictions to 1-based and reshape"""
        # Create empty array for all pixels
        full_output = np.zeros(original_shape[0] * original_shape[1], dtype = np.uint8)

        # Fill valid pixels (0-11 becomes 1-12)
        full_output[self.nan_mask] = predictions + 1

        return full_output.reshape(original_shape)

    def predict(self, input_raster: str, output_path: str) -> None:
        """Run prediction and save results"""
        if not os.path.exists(input_raster):
            raise FileNotFoundError(f"Input raster not found: {input_raster}")

        try:
            with rio.open(input_raster) as src:
                raster = src.read()
                profile = src.profile.copy()
                original_shape = (src.height, src.width)

                # Process data
                processed = self._preprocess_input(raster)
                preds = self.model.predict(processed) # type: ignore
                output = self._postprocess_output(preds, original_shape)

                # Save results
                profile.update(
                    dtype = rio.uint8,
                    count = 1,
                    nodata = 0,
                    compress = 'lzw'
                )

                os.makedirs(os.path.dirname(output_path), exist_ok = True)
                with rio.open(output_path, 'w', **profile) as dst:
                    dst.write(output, 1)

            print(f"Successfully saved XGBoost classification to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")

