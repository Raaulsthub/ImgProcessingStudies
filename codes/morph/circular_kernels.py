import cv2
import numpy as np
import matplotlib.pyplot as plt

# load and convert to binary
img = cv2.imread('./images/image.jpg', 0)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# disk kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# erosion
eroded = cv2.erode(binary, kernel)

# plot
fig, ax = plt.subplots(1, 3, figsize=(10, 10))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(binary, cmap='gray')
ax[1].set_title('Binary')
ax[2].imshow(eroded, cmap='gray')
ax[2].set_title('Eroded')
plt.show()
