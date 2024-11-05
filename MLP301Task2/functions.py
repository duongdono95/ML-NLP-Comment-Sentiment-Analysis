import cv2
import csv
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import (accuracy_score, auc, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt

# Function to preprocess image by enhancing contrast and smoothing
def imagePreprocessing(rgb_image):
  hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(hsv_image)
  # Set contrast and brightness adjustments for the saturation channel
  brightness = 0
  contrast = 1.2
  adjusted_s = cv2.addWeighted(s, contrast, np.zeros(s.shape, s.dtype), 0, brightness)
  # Apply Gaussian blur for noise reduction
  blurred_adjusted_s = cv2.GaussianBlur(adjusted_s, (1, 1), 0)
  return blurred_adjusted_s

# Function for binary segmentation of the image, isolating regions of interest
def imageSegmentation(input):
  threshold = 50
  _, binarised_input = cv2.threshold(input, threshold, 255, cv2.THRESH_BINARY)
  # Erode to reduce noise and refine segment edges
  kernel = np.ones((2, 2), np.uint8)
  eroded_output = cv2.erode(binarised_input, kernel, iterations=1)
  return eroded_output

# Function to calculate GLCM texture features from the segmented region
def GLCMCalculator(roi):
    # Calculate GLCM matrix with specific angles and distances
    g = graycomatrix(roi, [0, 1], [0, np.pi/2], levels=256)
    # Extract texture features and round for consistency
    contrast = round(float(np.mean(graycoprops(g, 'contrast'))), 2)
    energy = round(float(np.mean(graycoprops(g, 'energy'))), 2)
    homogeneity = round(float(np.mean(graycoprops(g, 'homogeneity'))), 2)
    correlation = round(float(np.mean(graycoprops(g, 'correlation'))), 2)
    dissimilarity = round(float(np.mean(graycoprops(g, 'dissimilarity'))), 2)
    ASM = round(float(np.mean(graycoprops(g, 'ASM'))), 2)
    return [contrast, dissimilarity, homogeneity, ASM, energy, correlation]
  
# Function to calculate HOG features from the grayscale region of interest
def HOGCalculator(gray_roi):
    resized_gray_roi = cv2.resize(gray_roi, (30, 30))
    # Extract HOG features and visualization
    features, hog_image = hog(
        resized_gray_roi,
        orientations = 9,
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2),
        visualize=True,
    )
    # Round feature values for consistency
    features = [round(float(f), 2) for f in features]
    return features

# Function for PCA transformation of feature lists, to reduce dimensions
def PCACalculator(features_list, n_component= 0.95, pca_model = None):
  name_list = [item[0] for item in features_list]
  label_list = [item[1] for item in features_list]
  features_list = [item[2] for item in features_list]
  # Standardize features to prepare for PCA
  scaler = StandardScaler()
  standardized_features_list = scaler.fit_transform(features_list)
  # Apply PCA: fit new model if none is provided, else use the provided model
  if not pca_model: 
    pca = PCA(n_components=n_component)
    PCA_HOG_features = pca.fit_transform(standardized_features_list)
  else: 
    pca = pca_model
    PCA_HOG_features = pca.transform(standardized_features_list)
  
  combined_list = []
  # Combine transformed features with image names and labels for easy export
  for name, label, pca_features in zip(name_list, label_list, PCA_HOG_features):
    rounded_pca_features = [round(float(f), 2) for f in pca_features]
    combined_list.append([name]  + [label] + [rounded_pca_features])
    
  return combined_list, pca

# Function to write features to a CSV file with a given header
def writeFeaturesToCSV(csv_name, row, header, main_folder, sub_folder):
    if not os.path.exists(main_folder):
      os.makedirs(main_folder)
    sub_folder_path = os.path.join(main_folder, sub_folder)
    
    if not os.path.exists(sub_folder_path):
      os.makedirs(sub_folder_path)
      
    file_exists = os.path.isfile(os.path.join(sub_folder_path, csv_name))
    with open(os.path.join(sub_folder_path, csv_name), mode="a", newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
          writer.writerow(header) 
        writer.writerow([row[0]] + [row[1]] + list(row[2]))
        
# Function to display performance metrics for a model     
def display_metrics(model_name, y_test, y_pred, y_test_binarized, y_score):
    print(f"Performance Metrics for {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    auc_score = roc_auc_score(y_test_binarized, y_score, average='weighted', multi_class="ovr")
    print(f"AUC Score: {auc_score:.2f}\n")

# Function to plot ROC curves for each class in a single model (multiclass)
def plot_roc_curves_multiclass(y_test_binarized, y_scores, model_name, n_classes):
    plt.figure(figsize=(10, 7))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} class {i+1} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves for {model_name}")
    plt.legend(loc="lower right")
    plt.show()

# Function to plot ROC curves for comparing multiple models
def plot_roc_curves_multimodel(y_test_binarized_list, y_scores_list, model_names, n_classes):
    plt.figure(figsize=(10, 7))
    
    for y_test_binarized, y_scores, model_name in zip(y_test_binarized_list, y_scores_list, model_names):
        fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_scores.ravel())
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparing Classifier Models")
    plt.legend(loc="lower right")
    plt.show()