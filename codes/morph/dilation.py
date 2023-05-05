import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/image.jpg', cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#  cross kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

# dilation
dilated = cv2.dilate(img, kernel, iterations=3)

# ploting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Image')
axs[1].imshow(dilated, cmap='gray')
axs[1].set_title('Dilation')
plt.show()
