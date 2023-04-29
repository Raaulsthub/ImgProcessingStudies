import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('./images/rhino.jpg', cv2.IMREAD_GRAYSCALE)

# create the kernel
kernel = np.array([[-1, 0, 1]])

# apply the gradient filter
filtered_img = cv2.filter2D(img, -1, kernel)


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(filtered_img, cmap='gray')
ax2.set_title('Filtered Image')
plt.show()

