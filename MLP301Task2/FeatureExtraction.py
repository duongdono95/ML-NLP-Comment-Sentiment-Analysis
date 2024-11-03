import cv2
import os
from functions import GLCMCalculator, imagePreprocessing, imageSegmentation, writeFeaturesToCSV, HOGCalculator, PCAHOGCalculator


GLCM_csv_filename = "GLCMFeatures.csv"
HOG_csv_filename = "HOGFeatures.csv"
PCA_HOG_csv_filename = "PCA_HOG_Features.csv"
combined_features_filename = "Combined_Features.csv"


def FeatureExtraction(image_dataset_path, features_main_folder, features_sub_folder ):
    GLCM_features_list = []
    HOG_features_list = []
    
    for label in os.listdir(image_dataset_path):
        label_path = os.path.join(image_dataset_path, label)
        if os.path.isdir(label_path) and not label.startswith("."):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                bgr_image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                
                preprocessed_image = imagePreprocessing(rgb_image)
                roi = imageSegmentation(preprocessed_image)

                GLCM_features = GLCMCalculator(roi)
                GLCM_features_list.append([image_name] + [label] + [GLCM_features] )

                HOG_features = HOGCalculator(roi)
                HOG_features_list.append([image_name] + [label] + [HOG_features] )

    PCA_HOG_features_list = PCAHOGCalculator(HOG_features_list)
    GLCM_features_header = ["Images","Label", "Contrast", "Dissimilarity", "Homogeneity", "ASM", "Energy", "Correlation" ]
    PCA_HOG_features_header = ["Images"] + ["Label"] + [f"Feature {i+1}" for i in range(len(PCA_HOG_features_list[0][2]))]
    combined_features_header = ["Images", "Label"] + ["Contrast", "Dissimilarity", "Homogeneity", "ASM", "Energy", "Correlation"] + [f"PCA HOG Feature {i+1}" for i in range(len(PCA_HOG_features_list[0][2]))]


    for GLCM_row in GLCM_features_list:
        writeFeaturesToCSV(GLCM_csv_filename, GLCM_row, GLCM_features_header, features_main_folder, features_sub_folder)
        
    for HOG_row in HOG_features_list:
        writeFeaturesToCSV(HOG_csv_filename, HOG_row, PCA_HOG_features_header, features_main_folder, features_sub_folder)
            
    for PCA_HOG_row in PCA_HOG_features_list:
        writeFeaturesToCSV(PCA_HOG_csv_filename, PCA_HOG_row, PCA_HOG_features_header, features_main_folder, features_sub_folder)

    if len(GLCM_features_list) != len(PCA_HOG_features_list):
        raise ValueError("The number of rows in GLCM_features_list and PCA_HOG_features_list are not equal.")
    else:
        for (a, b) in zip(GLCM_features_list, PCA_HOG_features_list):
            # Ensure matching image names and labels
            if a[0] == b[0] and a[1] == b[1]:
                # Combine features from GLCM and PCA HOG
                combined_features = a[2] + b[2]
                # Write to CSV
                writeFeaturesToCSV(combined_features_filename, [a[0], a[1], combined_features], combined_features_header, features_main_folder, features_sub_folder)
        