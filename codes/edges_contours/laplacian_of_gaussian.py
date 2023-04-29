import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/rhino.jpg', 0)

# laplacian of gaussian
img_log = cv2.GaussianBlur(img, (3, 3), 0)  # apply Gaussian filter
img_log = cv2.Laplacian(img_log, cv2.CV_64F)  # apply Laplacian filter
img_log = np.uint8(np.absolute(img_log))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(img_log, cmap='gray')
plt.title('Laplacian of Gaussian Edge Detection')
plt.show()
