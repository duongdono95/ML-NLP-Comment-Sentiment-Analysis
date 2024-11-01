import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def imagePreprocessing(rgb_image):
  hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  h, s, v = cv2.split(hsv_image)

  brightness = 0
  contrast = 1.2
  adjusted_s = cv2.addWeighted(s, contrast, np.zeros(s.shape, s.dtype), 0, brightness)

  blurred_adjusted_s = cv2.GaussianBlur(adjusted_s, (1, 1), 0)
  return blurred_adjusted_s

def imageSegmentation(input):
  threshold = 50
  _, binarised_input = cv2.threshold(input, threshold, 255, cv2.THRESH_BINARY)
  kernel = np.ones((2, 2), np.uint8)
  eroded_output = cv2.erode(binarised_input, kernel, iterations=1)
  return eroded_output

def GLCMCalculator(roi):
    g = graycomatrix(roi, [0, 1], [0, np.pi/2], levels=256)

    contrast = round(float(np.mean(graycoprops(g, 'contrast'))), 2)
    energy = round(float(np.mean(graycoprops(g, 'energy'))), 2)
    homogeneity = round(float(np.mean(graycoprops(g, 'homogeneity'))), 2)
    correlation = round(float(np.mean(graycoprops(g, 'correlation'))), 2)
    dissimilarity = round(float(np.mean(graycoprops(g, 'dissimilarity'))), 2)
    ASM = round(float(np.mean(graycoprops(g, 'ASM'))), 2)
    return [contrast, dissimilarity, homogeneity, ASM, energy, correlation]


def writeFeaturesToCSV(csv_name, row, header):
    file_exists = os.path.isfile(csv_name)
    with open(csv_name, mode="a", newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
          writer.writerow(header) 
        writer.writerow([row[0]] + list(row[1]) + [row[2]])