import os
import numpy as np
import joblib # type: ignore
import rasterio as rio # type: ignore
from typing import Tuple

class CatBoostLandcoverClassifier:
    """Predictor for a CatBoost model"""

    def __init__(self, model_path: str, scaler_path: str, num_classes: int = 12):
        """
        Initialise CatBoost predictor

        Args:
            - model_path (str): Path to CatBoost model (.joblib)
            - scaler_path (str): Path to corresponding scaler
            - num_classes (int): Number of landcover classes (default: 8)
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
        """Convert predictions to proper class numbering and reshape"""
        # Create empty array for all pixels
        full_output = np.zeros(original_shape[0] * original_shape[1], dtype = np.uint8)

        # Handle both 1D and 2D predictions
        if predictions.ndim > 1:
            # If probabilities or multiple outputs, take argmax for classification
            # For multi-class classification
            if predictions.shape[1] > 1:
                preds = np.argmax(predictions, axis = 1)
            # Binary classification
            else:
                preds = predictions[:, 0]
        else:
            preds = predictions

        # Check if class numbering requires adjustment
        min_class = np.min(preds)
        if min_class == 0:
            # Model uses 0-based classes (0-11), convert to 1-12
            adjusted_preds = preds + 1
        else:
            # Model already uses 1-based classes (1-12), use as-is
            adjusted_preds = preds

        # Fill valid pixels
        full_output[self.nan_mask] = adjusted_preds.astype(np.uint8)

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

            print(f"Successfully saved CatBoost classification to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")
