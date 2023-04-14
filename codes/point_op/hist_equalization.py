import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/tree.jpeg', cv2.IMREAD_GRAYSCALE)

# equalizing
img_eq = cv2.equalizeHist(img)

# plots
fig, axs = plt.subplots(3, 2, figsize=(10, 10))
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[1, 0].hist(img.flatten(), 256, [0, 256], color='r')
axs[1, 0].set_xlim([0, 256])
axs[1, 0].set_title('Original Histogram')
axs[0, 1].imshow(img_eq, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[1, 1].hist(img_eq.flatten(), 256, [0, 256], color='r')
axs[1, 1].set_xlim([0, 256])
axs[1, 1].set_title('Equalized Histogram')
axs[2, 0].plot(cv2.calcHist([img], [0], None, [256], [0, 256]).cumsum())
axs[2, 0].set_title('Original Cumulative Histogram')
axs[2, 1].plot(cv2.calcHist([img_eq], [0], None, [256], [0, 256]).cumsum())
axs[2, 1].set_title('Equalized Cumulative Histogram')
plt.show()