import cv2
import os
from functions import GLCMCalculator, imagePreprocessing, imageSegmentation, writeFeaturesToCSV, HOGCalculator, PCACalculator
import pickle

# Filenames for CSV files to store various feature types
GLCM_csv_filename = "GLCMFeatures.csv"
HOG_csv_filename = "HOGFeatures.csv"
PCA_HOG_csv_filename = "PCA_HOG_Features.csv"
Combined_features_filename = "Combined_Features.csv"
PCA_combined_feature_filename = "PCA_Combined_Features.csv"

# Function for extracting features from images in a given dataset
def FeatureExtraction(image_dataset_path, features_main_folder, features_sub_folder, PCA_model = None):
    # create feature main/ sub folders
    if not os.path.exists(features_main_folder):
        os.makedirs(features_main_folder)
    feature_sub_folder_path = os.path.join(features_main_folder, features_sub_folder)
    if not os.path.exists(feature_sub_folder_path):
        os.makedirs(feature_sub_folder_path)
    
    GLCM_features_list = []
    HOG_features_list = []
    combined_features_list = []
    # Loop over each label in the image dataset 
    for label in os.listdir(image_dataset_path):
        label_path = os.path.join(image_dataset_path, label)
        # Process only directories that represent labels (skip hidden files)
        if os.path.isdir(label_path) and not label.startswith("."):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                # Read the image and convert it from BGR to RGB
                bgr_image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                # Preprocess and segment the image to isolate the region of interest (ROI)
                preprocessed_image = imagePreprocessing(rgb_image)
                roi = imageSegmentation(preprocessed_image)
                # Extract GLCM features and add to the GLCM features list
                GLCM_features = GLCMCalculator(roi)
                GLCM_features_list.append([image_name] + [label] + [GLCM_features] )
                # Extract HOG features and add to the HOG features list
                HOG_features = HOGCalculator(roi)
                HOG_features_list.append([image_name] + [label] + [HOG_features] )
                # Combine GLCM and HOG features into a single feature vector for each image
                combined_features_list.append([image_name] + [label] + [GLCM_features + HOG_features])    
    # Perform PCA on combined features if no PCA model is provided, otherwise use the existing model
    if not PCA_model: 
        PCA_combined_features, pca = PCACalculator(combined_features_list, 100)
         # Save the trained PCA model
        with open(os.path.join(features_main_folder, "pca_model.pkl"), "wb") as f:
            pickle.dump(pca, f)
    else: 
        PCA_combined_features, pca = PCACalculator(combined_features_list, 100, PCA_model)
    # Use the provided PCA model to transform combined features
    GLCM_features_header = ["Images"] + ["Label"] + ["Contrast", "Dissimilarity", "Homogeneity", "ASM", "Energy", "Correlation" ]
    for GLCM_row in GLCM_features_list:
        writeFeaturesToCSV(GLCM_csv_filename, GLCM_row, GLCM_features_header, features_main_folder, features_sub_folder)
    # Define headers for HOG features and write to CSV file
    HOG_features_header = ["Images"] + ["Label"] + [f"Feature {i+1}" for i in range(len(HOG_features_list[0][2]))] 
    for HOG_row in HOG_features_list:
        writeFeaturesToCSV(HOG_csv_filename, HOG_row, HOG_features_header, features_main_folder, features_sub_folder)
    # Define headers for combined features (GLCM + HOG) and write to CSV file    
    combined_features_header = ["Images"] + ["Label"] + ["Contrast", "Dissimilarity", "Homogeneity", "ASM", "Energy", "Correlation" ] + [f"Feature {i+1}" for i in range(len(HOG_features_list[0][2]))] 
    for combined_row in combined_features_list:
        writeFeaturesToCSV(Combined_features_filename, combined_row, combined_features_header, features_main_folder, features_sub_folder)
    # Define headers for PCA-combined features and write to CSV file
    PCA_combined_features_header = ["Images"] + ["Label"] + [f"Feature {i+1}" for i in range(len(PCA_combined_features[0][2]))] 
    for PCA_combined_row in PCA_combined_features:
        writeFeaturesToCSV(PCA_combined_feature_filename, PCA_combined_row, PCA_combined_features_header, features_main_folder, features_sub_folder)
    