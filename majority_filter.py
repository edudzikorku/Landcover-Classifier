import rasterio as rio # type: ignore
import numpy as np # type: ignore
import config


input_raster = "LANDCOVER.tif"
output_raster = "LANDCOVER_Majority_4_by_4.tif"


def majority_filter(input_path: str, output_path: str, window_size: int = 3) -> None:
    """
    Majority filter with custom window size.
    
    Parameters:
    -----------
        - input_path (str): Path to input raster
        - output_path (str): Path to output raster
        - window_size (int): Size of square window (3 = 3 x 3, 5 = 5 x 5, etc.)
    """
    
    with rio.open(input_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        
        # Apply the filter
        filtered = apply_filter(data, window_size)
        
        # Write result
        with rio.open(output_path, 'w', **profile) as dst:
            dst.write(filtered, 1)

def apply_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply majority filter to the data array.

    Parameters:
    -----------
        - data (np.ndarray): Input data array
        - window_size (int): Size of square window (3 = 3 x 3, 5 = 5 x 5, etc.)
    """
    result = data.copy()
    rows, cols = data.shape
    offset = window_size // 2
    
    for i in range(offset, rows - offset):
        for j in range(offset, cols - offset):
            # Extract window
            window = data[i - offset: i + offset + 1, j - offset: j + offset + 1]
            
            # Find majority value
            values, counts = np.unique(window, return_counts = True)
            majority_idx = np.argmax(counts)
            result[i, j] = values[majority_idx]
    
    print(f"Applied majority filter with window size: {window_size}x{window_size}.")
    print(f"Output saved at: {output_raster}")
    
    return result

# Usage
if __name__ == "__main__":
    majority_filter(input_raster, output_raster, window_size = 4)
 