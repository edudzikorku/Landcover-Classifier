import pandas as pd #type: ignore
import numpy as np #type: ignore
import warnings #type: ignore
import seaborn as  sns #type: ignore
from typing import Dict #type: ignore
from scipy.linalg import inv #type: ignore
import matplotlib.pyplot as plt #type: ignore

warnings.filterwarnings("ignore")

class JeffriesMatusitaDistance:
    def __init__(self, df, spectral_columns, class_names = Dict[int, str], class_column: str = 'landcover') -> None:
        """
        Initialize the analysis with a dataset

        Parameters:
            - df: pandas DataFrame containing the spectral data and class labels
            - spectral_columns: list of column names containing spectral/index values
            - class_column: name of the column containing class labels
        """
        # Create a copy to avoid modifying original data
        self.df = df.copy()  
        self.spectral_columns = spectral_columns
        self.class_column = class_column
        self.class_names = class_names

        self.jm_matrix = None

        # Convert spectral columns to float
        self._validate_and_convert_data()

    def _validate_and_convert_data(self) -> None:
        """
        Validate and convert data types of spectral columns to float64
        """
        # Convert spectral columns to float
        for col in self.spectral_columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors = 'coerce')
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")
                print(f"Column {col} contains values: {self.df[col].unique()}")

        # Drop rows with NaN values
        self.df = self.df.dropna(subset = self.spectral_columns)

        # Ensure class column is integer
        self.df[self.class_column] = self.df[self.class_column].astype(int)

        # Validate that we have data for analysis
        if self.df.empty:
            raise ValueError("No valid data remaining after cleaning")

    def _calculate_jm_distance(self, class1_data, class2_data) -> float:
        """
        Calculate Jeffries-Matusita distance between two classes
        """
        # Ensure data is float64
        class1_data = class1_data.astype(np.float64)
        class2_data = class2_data.astype(np.float64)

        # Calculate means
        mean1 = np.mean(class1_data, axis = 0)
        mean2 = np.mean(class2_data, axis = 0)

        # Calculate covariance matrices
        cov1 = np.cov(class1_data, rowvar = False)
        cov2 = np.cov(class2_data, rowvar = False)

        # Calculate average covariance matrix
        cov_avg = (cov1 + cov2) / 2

        try:
            # Calculate Bhattacharyya distance
            mean_diff = mean1 - mean2
            inv_cov_avg = inv(cov_avg)
            term1 = (1/8) * np.dot(np.dot(mean_diff.T, inv_cov_avg), mean_diff)

            term2 = (1/2) * np.log(np.linalg.det(cov_avg) /
                                  np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))

            b_distance = term1 + term2

            # Convert to JM distance
            jm_distance = np.sqrt(2 * (1 - np.exp(-b_distance)))

            return jm_distance

        except np.linalg.LinAlgError:
            return np.nan

    def compute_pairwise_distances(self) -> pd.DataFrame:
        """
        Compute pairwise JM distances between all land cover classes
        """
        classes = sorted(self.df[self.class_column].unique())
        jm_matrix = pd.DataFrame(index = classes, columns = classes, dtype = np.float64)

        for i in classes:
            for j in classes:
                if i <= j:
                    class1_data = self.df[self.df[self.class_column] == i][self.spectral_columns].values
                    class2_data = self.df[self.df[self.class_column] == j][self.spectral_columns].values

                    if i == j:
                        jm_matrix.loc[i, j] = 0
                    else:
                        jm_distance = self._calculate_jm_distance(class1_data, class2_data)
                        jm_matrix.loc[i, j] = jm_distance
                        jm_matrix.loc[j, i] = jm_distance

        # Convert class numbers to names
        jm_matrix.index = [self.class_names[i] for i in jm_matrix.index]
        jm_matrix.columns = [self.class_names[i] for i in jm_matrix.columns]

        self.jm_matrix = jm_matrix
        return jm_matrix

    def plot_heatmap(self, figsize = (12, 10), cmap = 'YlOrRd', annot = True, 
                     fmt = '.3f', save_path = None) -> None:
        """
        Plot a heatmap of the JM distances
        """
        if self.jm_matrix is None:
            self.compute_pairwise_distances()

        # Convert matrix to float64 to ensure compatibility
        plot_matrix = self.jm_matrix.astype(np.float64)

        plt.figure(figsize = figsize)
        try:
          sns.heatmap(plot_matrix,
                    annot = annot,
                    cmap = cmap,
                    fmt = fmt,
                    square = True,
                    vmin = 0,
                    vmax = 2)

          plt.title('Jeffries-Matusita Distance Between Land Cover Classes',
                  pad = 20,
                  size = 14)

          if save_path:
              plt.savefig(save_path,
                        bbox_inches = 'tight',
                        dpi = 300)
          plt.show()

        except Exception as e:
          print(f"Error plotting heatmap: {str(e)}")

    def save_matrix(self, filepath) :
        """
        Save the JM distance matrix to a CSV file
        """
        if self.jm_matrix is None:
            self.compute_pairwise_distances()
        self.jm_matrix.round(3).to_csv(filepath)

    def get_separability_summary(self):
        """
        Generate a summary of class separability
        """
        if self.jm_matrix is None:
            self.compute_pairwise_distances()

        summary = {
            'excellent': [],  # JM > 1.9
            'moderate': [],   # 1.0 < JM <= 1.9
            'poor': []        # JM <= 1.0
        }

        for i in range(len(self.jm_matrix.index)):
            for j in range(i + 1, len(self.jm_matrix.columns)):
                class1 = self.jm_matrix.index[i]
                class2 = self.jm_matrix.columns[j]
                jm_value = float(self.jm_matrix.iloc[i, j])

                pair = f"{class1} vs {class2}"
                if jm_value > 1.9:
                    summary['excellent'].append((pair, jm_value))
                elif jm_value > 1.0:
                    summary['moderate'].append((pair, jm_value))
                else:
                    summary['poor'].append((pair, jm_value))

        return summary