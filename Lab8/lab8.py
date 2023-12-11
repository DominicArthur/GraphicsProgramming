# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define the number of rows and columns for subplots
nrows = 2
ncols = 2

# Read the original image from file
imgOrig = cv2.imread("ATU.jpg")

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Apply a 3x3 Gaussian blur to the grayscale image
img3 = cv2.GaussianBlur(imgGray, (3, 3), 0)

# Apply a 13x13 Gaussian blur to the grayscale image
img13 = cv2.GaussianBlur(imgGray, (13, 13), 0)

# Plot the original image using matplotlib
plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the grayscale image using matplotlib
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the 3x3 Gaussian blur image
plt.subplot(nrows, ncols, 3), plt.imshow(img3, cmap='gray')
plt.title('3x3 Gaussian Blur'), plt.xticks([]), plt.yticks([])

# Plot the 13x13 Gaussian blur image
plt.subplot(nrows, ncols, 4), plt.imshow(img13, cmap='gray')
plt.title('13x13 Gaussian Blur'), plt.xticks([]), plt.yticks([])

plt.show()  # Display the plot

