import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 1
ncols = 1

imgOriginal = cv2.imread('ATU1.jpg')
imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

plt.subplot(nrows, ncols, 1), plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.show()
