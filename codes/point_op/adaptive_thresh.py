import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image in gray
img = cv2.imread('./images/tree.jpeg', 0)

# gausian adaptive threshold
gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# median adaptive threshold
median = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

#plot
fig, axs = plt.subplots(1, 3, figsize=(10, 10))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(gaussian, cmap='gray')
axs[1].set_title('Gaussian')
axs[2].imshow(median, cmap='gray')
axs[2].set_title('Median')
plt.show()
