import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('./images/cat.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# box kernel
kernel = np.ones((3,3), np.uint8)
# gradient operation
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

# plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')
axs[1].imshow(gradient, cmap='gray')
axs[1].set_title('Morphological Gradient')
plt.show()