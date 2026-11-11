import os 
import joblib # type: ignore
import numpy as np # type: ignore
import rasterio as rio # type: ignore
from typing import Tuple


class RFLandcoverClassifier:
    """
    Random Forest land cover classifier with integrated scaler support
    
    Args:
        - model_path (str): Path to trained RF model (.joblib)
        - scaler_path (str): Path to corresponding feature scaler
        - num_classes (int): Number of expected output classes (default: 12)
        - class_offset (int): Value to add to predicted classes (default: 0)
    """

    def __init__(self, model_path: str, scaler_path: str, num_classes: int = 12, class_offset: int = 0) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.num_classes = num_classes
        self.class_offset = class_offset
        self.model = None
        self.scaler = None
        self.nan_mask = None
        self._load_assets()

    def _load_assets(self) -> None:
        """Load both the Random Forest model and its scaler"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model/scaler: {str(e)}")

    def _preprocess_input(self, raster: np.ndarray) -> np.ndarray:
        """
        Prepare raster data for prediction:
        1. Reshape to 2D (pixels × bands)
        2. Filter out NaN values
        3. Apply feature scaling
        
        Args:
            raster (np.ndarray): Input raster (bands, height, width)
            
        Returns:
            np.ndarray: Scaled, NaN-free data ready for prediction
        """
        # Reshape to (height * width, bands)
        reshaped = raster.transpose(1, 2, 0).reshape(-1, raster.shape[0])
        
        # Create mask for valid pixels
        self.nan_mask = ~np.isnan(reshaped).any(axis = 1)
        valid_data = reshaped[self.nan_mask]
        
        return self.scaler.transform(valid_data) # type: ignore

    def _postprocess_output(self, predictions: np.ndarray, 
                          shape: Tuple[int, int]) -> np.ndarray:
        """
        Process model predictions:
        1. Apply class offset
        2. Reshape to original dimensions
        3. Restore NaN positions
        
        Args:
            predictions (np.ndarray): Model predictions (1D array)
            shape (Tuple): Original raster dimensions (height, width)
            
        Returns:
            np.ndarray: 2D classification map with proper values and shape
        """
        output = np.zeros(shape[0] * shape[1], dtype = np.uint8)
        output[self.nan_mask] = predictions + self.class_offset
        return output.reshape(shape)

    def predict(self, input_raster: str, output_path: str) -> None:
        """
        Perform classification and save results to GeoTIFF
        
        Args:
            - input_raster (str): Path to input raster file
            - output_path (str): Path to save output classification
            
        Raises:
            - FileNotFoundError: If input file doesn't exist
            - RuntimeError: If prediction fails
        """
        if not os.path.exists(input_raster):
            raise FileNotFoundError(f"Input raster not found: {input_raster}")

        try:
            with rio.open(input_raster) as src:
                # Read data and get metadata
                raster = src.read()
                profile = src.profile.copy()
                height, width = src.height, src.width

                # Process and predict
                processed = self._preprocess_input(raster)
                preds = self.model.predict(processed) # type: ignore
                classification = self._postprocess_output(preds, (height, width))

                # Configure output
                profile.update(
                    dtype = rio.uint8,
                    count = 1,
                    nodata = 0,
                    compress = 'lzw'
                )

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok = True)

                # Save results
                with rio.open(output_path, 'w', **profile) as dst:
                    dst.write(classification, 1)

            print(f"Successfully saved RF classification to {output_path}")

        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")