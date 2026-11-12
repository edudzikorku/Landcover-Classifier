### Introduction and Problem
To effectively address the rapid loss of wetlands and forests, state-of-the-art tools and approaches are required. Earth Observation (EO) and Geographic Information Systems (GIS) technologies offer a means of cost-effective, large-scale, and near-accurate monitoring and assessment of wetland changes. A variety of data sources, including radar, LiDAR, very high resolution (VHR), medium resolution optical, and coarse EO products, have been used in previous studies to monitor wetland changes effectively. Frequently utilised products include those from the Landsat satellite series, Sentinel-2ASentinel-1 Synthetic Aperture Radar (SAR), WorldView-2/3, and RapidEye. Likewise, some studies have combined optical, radar, and LiDAR data to overcome challenges posed by cloud cover in optical images, particularly in tropical regions.

Recent advancements in geo-computing have transformed the fields of EO and environmental monitoring by providing high levels of accuracy and efficiency in mapping and monitoring tools. Combining EO and GIS techniques with Machine Learning (ML) and Deep Learning (DL) models has proven effective in improving the accuracy of wetland and forest classification, mapping, and temporal change detection. As a result, these techniques have become increasingly important in environmental research by providing data-driven insights that support more effective conservation and management strategies. Commonly used models in this field include Random Forests (RF), Support Vector Machines (SVM), Extreme Gradient Boosting (XGBoost), Decision Trees (DT), Boosted Decision Trees (BDT), k-Nearest Neighbour (k-NN) and Convolutional Neural Networks (CNNs), as well as several of their variants.

However, mastering and applying these techniques to classification tasks requires a steep learning curve and major time commitments. In response, researchers have made major contributions by implementing these classification techniques across open-source and commercial software products. Notably, the Semi-automatic classification plugin in QGIS and supervised classification workflows in ArcGIS have abstracted the technical details to allow practitioners to focus on data preparation and final classification. 

Despite these efforts, much of the effort still resides with the analyst. For instance, they are responsible for manual feature selection, manual model parameter fine-tuning, and are also not able to run and compare multiple classification models’ performance concurrently. Therefore, the current classification toolbox is designed to reduce the level of effort on the part of practitioners by allowing them to perform less feature selection and fine-tuning and on-the-fly model comparison. Additionally, the tool implements a K-Fold cross-validation (CV) technique to ensure the models are reliable across different data splits.

### How to Use
The toolbox is designed for use in ArcGIS Pro. The following instructions serve as a guide to help users test the toolbox in ArcGIS Pro:

1. Clone your default ArcGIS Pro environment. Refer to this <a href=https://doc.esri.com/en/arcgis-pro/latest/arcpy/get-started/clone-an-environment.html target=_blank width=150>blog post</a> for guidance on how to perform this task. Choose any name for your new environment that suits your needs.

   
2. Visit the <a href=https://github.com/edudzikorku/Landcover-Classifier.git target=_blank>GitHub repository</a> of the toolbox and download the entire repository.
<img src=images-for-instructions/download-from-source.png/>

3. In your download directory, extract the contents of the compressed file.

4. Open the Python Command Prompt and launch and activate your cloned conda environment.
<img src=images-for-instructions/new-environment.png/>

5. Navigate to your download directory, locate the requirements.txt file, and copy its contents.
<img src=images-for-instructions/requirements.png/>
6. Move to the activated environment in the command prompt and install all packages using either pip or conda. For instance, you can run the following command with pip:  
   ``` bash
    pip install "numpy>=1.21.0" "pandas>=1.3.0" "scikit-learn<=1.8.0" "rasterio>=1.2.0" "geopandas>=0.10.0" "xgboost>=1.5.0" "catboost>=1.0.0" "imbalanced-learn>=0.9.0" "matplotlib>=3.5.0" "seaborn>=0.11.0" "scipy>=1.7.0" "tqdm>=4.62.0" "joblib>=1.1.0".
   ```

7. After all packages have been installed, launch ArcGIS Pro and create a new project.

8. In your Catalog pane, right-click on ‘Toolboxes’ and select ‘Add Toolbox’.
<img src=images-for-instructions/add-toolbox.png/>

9. Navigate to your download directory and select the ‘LandCoverClassifier.pyt’ file.
<img src=images-for-instructions/select-toolbox.png/>

10. When prompted, select ‘Yes’ to proceed.
<img src=images-for-instructions/access-permission.png/>

11. Congratulations! The toolbox is now ready for use.

### Script Tools

1. **Train Landcover Model:** This tool takes an input multiband raster and a feature class containing target Land Cover / Land Use (LULC) classes. It trains up to three classification algorithms based on user preference. After training, the selected model(s) is/are saved for downstream tasks.

2. **Predict Landcover:** This tool takes the trained model from step 1 and applies it across the entire image to perform pixel-based classification.

3. **Spectral Separability Analysis:** This tool implements the Jeffries-Matusita (JM) distance to validate spectral separability between target classes. The output guides the analyst regarding the quality of training samples collected for classification.

4. **Apply Majority Filter:** This is a post-classification technique used to smooth the classification from step 2 and reduce classification noise.
