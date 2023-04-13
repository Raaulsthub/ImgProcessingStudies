import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./images/sunset.jpg', 0)

# adaptive histogram
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)

# plot
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(clahe_img, cmap='gray')
axs[1].set_title('Adaptive Histogram Equalized Image')

plt.show()