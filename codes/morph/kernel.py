import cv2
import numpy as np
import matplotlib.pyplot as plt


# load and limiarization
img = cv2.imread('./images/image.jpg', cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# structuring elements
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

# erosion and dilation with cross kernel
dilation_cross = cv2.dilate(img, cross_kernel, iterations=3)
erosion_cross = cv2.erode(img, cross_kernel, iterations=3)

# erosion and dilation with block kernel
dilation_block = cv2.dilate(img, block_kernel, iterations=3)
erosion_block = cv2.erode(img, block_kernel, iterations=3)

# ploting images
fig, axs = plt.subplots(nrows=3, ncols=2)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 1].imshow(img, cmap='gray')
axs[0, 1].set_title('Original')
axs[1, 0].imshow(dilation_cross, cmap='gray')
axs[1, 0].set_title('Dilation (cross)')
axs[1, 1].imshow(erosion_cross, cmap='gray')
axs[1, 1].set_title('Erosion (cross)')
axs[2, 0].imshow(dilation_block, cmap='gray')
axs[2, 0].set_title('Dilation (block)')
axs[2, 1].imshow(erosion_block, cmap='gray')
axs[2, 1].set_title('Erosion (block)')
fig.subplots_adjust(hspace=0.5)
plt.show()
