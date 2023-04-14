import cv2
import numpy as np
from matplotlib import pyplot as plt

# loading reference image and image
ref_img = cv2.imread('./images/tree.jpeg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('./images/deer.jpeg', cv2.IMREAD_GRAYSCALE)

# histogram of both images calculated
hist_img, bins_img = np.histogram(img.flatten(), 256, [0, 256])
hist_ref, bins_ref = np.histogram(ref_img.flatten(), 256, [0, 256])

# histograms cummulative distribution function
cdf_img = hist_img.cumsum()
cdf_img_normalized = cdf_img * hist_img.max() / cdf_img.max()
cdf_ref = hist_ref.cumsum()
cdf_ref_normalized = cdf_ref * hist_ref.max() / cdf_ref.max()

# creating map table
mapping_table = np.zeros(256, dtype=np.uint8)
for i in range(256):
    j = 0
    while j < 256 and cdf_ref_normalized[j] < cdf_img_normalized[i]:
        j += 1
    mapping_table[i] = j

# applying the table to change the original image
equalized_img = cv2.LUT(img, mapping_table)

# plotting images and histograms
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 1].imshow(ref_img, cmap='gray')
axs[0, 1].set_title('Reference Image')
axs[0, 2].imshow(equalized_img, cmap='gray')
axs[0, 2].set_title('Equalized Image')
axs[1, 0].plot(hist_img)
axs[1, 0].set_title('Original Image Histogram')
axs[1, 1].plot(hist_ref)
axs[1, 1].set_title('Reference Image Histogram')
axs[1, 2].plot(cv2.calcHist([equalized_img], [0], None, [256], [0, 256]))
axs[1, 2].set_title('Equalized Image Histogram')
plt.show()
