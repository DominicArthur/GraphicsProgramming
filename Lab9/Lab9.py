# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Define the number of rows and columns for subplots
nrows = 2
ncols = 3

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

# ORB
imgOrb = imgGray.copy()

# Create an ORB object
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
kp = orb.detect(imgOrb, None)
kp, des = orb.compute(imgOrb, kp)

# Draw keypoints on the image
imgOrb = cv2.drawKeypoints(imgOrb, kp, None, color=(0, 255, 0), flags=0)

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

# Plot the original image
plt.subplot(nrows, ncols, 1), plt.imshow(cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Plot the grayscale image using matplotlib
plt.subplot(nrows, ncols, 2), plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the Harris corners image
plt.subplot(nrows, ncols, 3), plt.imshow(imgHarris, cmap='gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the ShiTomasi corners image
plt.subplot(nrows, ncols, 4), plt.imshow(imgShiTomasi, cmap='gray')
plt.title('ShiTomasi'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

# Plot the ORB corners image
plt.subplot(nrows, ncols, 5), plt.imshow(imgOrb, cmap='gray')
plt.title('ORB'), plt.xticks([]), plt.yticks([])  # Set title and remove tick marks

plt.show()  # Display the plot



