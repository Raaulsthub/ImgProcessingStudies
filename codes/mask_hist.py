import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/sunset.jpg', 0)

# mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[200:400, 300:500] = 255

# histogram based on mask
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

# plt image, mask, histogram
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Grayscale Image')

ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Mask')

ax[2].plot(hist_mask)
ax[2].set_title('Histogram of Gray Image based on Mask')

plt.show()
