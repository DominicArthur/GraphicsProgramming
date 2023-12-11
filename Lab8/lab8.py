# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define the number of rows and columns for subplots
nrows = 2
ncols = 1

# Read the original image from file
imgOrig = cv2.imread("ATU.jpg")

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Plot the original image using matplotlib
plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the grayscale image using matplotlib
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

plt.show()  # Display the plot
