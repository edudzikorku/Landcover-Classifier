import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import rasterio as rio  # type: ignore
import geopandas as gpd  # type: ignore
from rasterio.mask import mask # type: ignore
from typing import List, Tuple, Dict, Optional
from config import TARGET_COLUMN, TARGET_COLUMN_DESC
from sklearn.impute import SimpleImputer # type: ignore
from shapely.geometry import Point # type: ignore
from rasterio.warp import calculate_default_transform, reproject, Resampling # type: ignore


os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'


class DataProcessor:
    """Handles all raster and vector data processing operations."""

    def __init__(self, raster_path: str, vector_path: str) -> None:
        """
        Initialises the DataProcessor class with input paths.

        Args:
            - raster_path (str): Path to the raster file.
            - vector_path (str): Path to the vector file.
        """
        self.raster_path = raster_path
        self.vector_path = vector_path
        self.gdf = None
        self.raster_profile = None
        self.raster_shape = None
        self.original_raster = None

    def create_directories(self, directories: List[str]) -> None:
        """
        Create necessary directories for storing files

        Args:
        - directories (List[str]): List of directories to create.

        Returns:
            None
        """
        for directory in directories:
            os.makedirs(directory, exist_ok = True)


    def reproject_raster(self, output_path: str, dst_crs: str, location: str) -> None:
        """
        Reprojects a raster file to a specified coordinate reference system (CRS).

        Args:
            - output_path (str): Path to the output raster file.
            - dst_crs (str): Target CRS in the form of a string.
            - location (str): Location identifier for output file name.

        Returns:
            None
        """
        with rio.open(self.raster_path) as src:
            # Get the relevant info
            transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)

            # Copy the metadata info of th original raster
            kwargs = src.meta.copy()

            # Update the metadata with the new CRS and transform info
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            # Specify the output file name
            output_file = os.path.join(output_path, f"{location}_REPROJ.tif")

            # Reproject and save the output_file
            with rio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source = rio.band(src, i),
                        destination = rio.band(dst, i),
                        src_transform = src.transform,
                        src_crs = src.crs,
                        dst_transform = transform,
                        dst_crs = dst_crs,
                        resampling = Resampling.nearest)

            self.raster_path = output_file

    def extract_points_by_lulc(self, lulc_specs: Dict[int, int]) -> gpd.GeoDataFrame:
        """
        Extract random points within polygons for each LULC class

        Args:
            lulc_specs: Dictionary mapping class values to number of points
                        e.g., {1: 500, 2: 600}

        Returns:
            GeoDataFrame containing the generated points
        """
        gdf = gpd.read_file(self.vector_path)
        points_list = []

        for lulc_value, num_points in lulc_specs.items():
            lulc_polygons = gdf[gdf[TARGET_COLUMN] == lulc_value]

            if len(lulc_polygons) == 0:
                print(f"Warning: No polygons found for LULC value {lulc_value}")
                continue

            points_per_polygon = num_points // len(lulc_polygons)
            extra_points = num_points % len(lulc_polygons)

            for idx, row in lulc_polygons.iterrows():
                this_polygon_points = points_per_polygon
                if extra_points > 0:
                    this_polygon_points += 1
                    extra_points -= 1

                minx, miny, maxx, maxy = row.geometry.bounds
                points_generated = 0
                attempts = 0
                max_attempts = this_polygon_points * 10

                while points_generated < this_polygon_points and attempts < max_attempts:
                    point = Point(
                        np.random.uniform(minx, maxx),
                        np.random.uniform(miny, maxy))

                    if row.geometry.contains(point):
                        points_list.append({
                            'geometry': point,
                            'Value': row[TARGET_COLUMN],
                            'Class': row[TARGET_COLUMN_DESC]
                        })
                        points_generated += 1
                    attempts += 1

        points_gdf = gpd.GeoDataFrame(points_list, crs = gdf.crs)
        return points_gdf

    def extract_raster_values(self, points_gdf: gpd.GeoDataFrame) -> None:
      """
      Extract raster pixel values to points.
      Args:
          gdf (GeoDataFrame): GeoDataFrame with points.
          input_raster_path (str): Path to the raster.
      Returns:
      GeoDataFrame with raster values.
      """
      # Access raster
      with rio.open(self.raster_path) as src:
          # Ensure the CRS of the vector data matches the raster
          if points_gdf.crs != src.crs:
              points_gdf = points_gdf.to_crs(src.crs)

          # Get the band count and band names from metadata
          band_count = src.count
          band_names = [src.descriptions[i] if src.descriptions[i] else f'B{i+1}' for i in range(band_count)]

          # Prepare a list to store the pixel values
          pixel_values = []

          # Get raster bounds
          bounds = src.bounds

          # Iterate over each point in the vector data
          for geom in points_gdf.geometry:
              if geom.x < bounds.left or geom.x > bounds.right or geom.y < bounds.bottom or geom.y > bounds.top:
                  # Point is outside the raster bounds
                  pixel_values.append([np.nan] * band_count)
              else:
                  # Get the roW and col of the point
                  row, col = src.index(geom.x, geom.y)

                  # Initialize a list to store pixel values for each band
                  values = []
                  for i in range(1, band_count + 1):
                      # Read the pixel value for the current band at the point location
                      band_value = src.read(i, window = ((row, row + 1), (col, col + 1)))
                      values.append(band_value[0, 0])

                  pixel_values.append(values)

          # Convert the list of pixel values to a DataFrame
          pixel_df = pd.DataFrame(pixel_values, columns = band_names)

      # Combine the pixel values DataFrame with the GeoDataFrame
      points_gdf = points_gdf.reset_index(drop = True)
      points_gdf = pd.concat([points_gdf, pixel_df], axis = 1)

      return points_gdf

    def extract_polygon_values(self, lulc_specs: Optional[Dict[int, int]] = None,
                               class_field: str = None, class_name_field: str = None) -> gpd.GeoDataFrame:  # type: ignore
      """
      Extract all pixel values within polygons for each LULC class

      Args:
          -lulc_specs: Optional dictionary to limit number of polygons per class
          -class_field: Field name for LULC class values
          -class_name_field: Field name for LULC class names
          
      Returns:
          GeoDataFrame containing pixel values with class labels
      """
      if class_field is None:
          class_field = TARGET_COLUMN

      if class_name_field is None:
          class_name_field = TARGET_COLUMN_DESC

      gdf = gpd.read_file(self.vector_path)
      points_list = []

      with rio.open(self.raster_path) as src:
          gdf = gdf.to_crs(src.crs)
          band_count = src.count
          band_names = [src.descriptions[i] if src.descriptions[i] else f'B{i+1}' 
                      for i in range(band_count)]
          
          # Filter polygons if specs provided
          if lulc_specs:
              gdf = gdf[gdf[TARGET_COLUMN].isin(lulc_specs.keys())]
              
          for idx, row in gdf.iterrows():
              try:
                  # Mask the raster with current polygon
                  masked, transform = mask(src, [row.geometry], crop = True, all_touched = True)
                  
                  # Reshape to (pixels, bands)
                  pixels = masked.reshape(band_count, -1).T
                  
                  # Create records for each pixel
                  for pixel in pixels:

                      record = {
                          class_field: row[class_field]
                        #   'Class': row[TARGET_COLUMN_DESC],
                        #   'geometry': row.geometry.centroid
                      }
                      
                      if class_name_field and class_name_field in row:
                          record[class_name_field] = row[class_name_field]

                      # Add band values
                      record.update(dict(zip(band_names, pixel)))
                      points_list.append(record)
                      
              except Exception as e:
                  print(f"Error processing polygon {idx}: {str(e)}")
      
    #   return gpd.GeoDataFrame(points_list, geometry = 'geometry', crs = gdf.crs)
      return pd.DataFrame(points_list)

    def rename_columns(self, column_mapping: Dict[str, str]) -> None:
        """
        Renames the columns of the dataframe based on column mapping.

        Args:
            - column_mapping (Dict[str, str]): A dictionary mapping the old column names to the new column names.

        Returns:
            - None
        """
        self.gdf.rename(columns = column_mapping, inplace = True) # type: ignore

    def handle_missing_values(self, X: np.ndarray, strategy: str = 'median') -> np.ndarray:
        """
        Handle missing values in feature matrix.
        
        Args:
            - X: Feature matrix
            - strategy: Strategy to handle missing values ('mean', 'median', 'most_frequent', 'zero')
            
        Returns:
            - Feature matrix with missing values handled
        """
        if strategy == 'zero':
            X = np.nan_to_num(X, nan = 0.0)
        else:
            imputer = SimpleImputer(strategy = strategy)
            X = imputer.fit_transform(X)
        
        return X
    def get_features_and_targets(self,
                                feature_columns: List[str],
                                target_column: str,
                                geodataframe: gpd.GeoDataFrame,
                                imputation_strategy: str = 'median'
                                ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoSeries, np.ndarray, np.ndarray]:
        """
        Splits the dataframe into features and targets based on the specified columns.

        Args:
            - feature_columns (List[str]): A list of column names to be used as features.
            - target_column (str): The column name to be used as the target.
            - geodataframe (gpd.GeoDataFrame): The input GeoDataFrame.
            - imputation_strategy (str): Strategy to handle missing values in features.


        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoSeries, np.ndarray, np.ndarray]:
                A tuple containing:
                - The full GeoDataFrame.
                - The GeoDataFrame containing the feature columns.
                - The GeoSeries containing the target column.
                - A numpy array of feature values.
                - A numpy array of target values.
        """
        # Extract features and target
        features_gdf = geodataframe[feature_columns] # type: ignore
        target_series = geodataframe[target_column] # type: ignore

        # Convert to numpy arrays
        features_array = features_gdf.values # type: ignore
        target_array = target_series.values # type: ignore
        # Handle missing values in features
        features_array = self.handle_missing_values(features_array, strategy = imputation_strategy)

        return geodataframe, features_gdf, target_series, features_array, target_array