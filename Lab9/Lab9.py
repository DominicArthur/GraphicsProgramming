# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define the number of rows and columns for subplots
nrows = 3
ncols = 1

# Read the original image from file
imgOriginal = cv2.imread('ATU1.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

# Harris corners
imgHarris = imgGray.copy()
harrisCorners = cv2.cornerHarris(imgHarris, 2, 3, 0.04)
harrisCorners = cv2.dilate(harrisCorners, None)

# Shi Tomasi
imgShiTomasi = imgGray.copy()
corners = cv2.goodFeaturesToTrack(imgShiTomasi, 80, 0.01, 10)
corners = np.int64(corners)

# Threshold for harris corners
threshold = 0.01 * harrisCorners.max()

# Iterate through all corners in Harris
for i in range(harrisCorners.shape[0]):
    for j in range(harrisCorners.shape[1]):
        if harrisCorners[i, j] > threshold:
            cv2.circle(imgHarris, (j, i), 2, (0, 255, 0), -1)

# Iterate through all corners in ShiTomasi
for i in corners:
    x, y = i.ravel()
    cv2.circle(imgShiTomasi, (x, y), 4, 255, -1)

# Plot the grayscale image using matplotlib
plt.subplot(nrows, ncols, 1), plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the Harris corners image
plt.subplot(nrows, ncols, 2), plt.imshow(imgHarris, cmap='gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the ShiTomasi corners image
plt.subplot(nrows, ncols, 3), plt.imshow(imgShiTomasi, cmap='gray')
plt.title('ShiTomasi'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

plt.show()  # Display the plot


