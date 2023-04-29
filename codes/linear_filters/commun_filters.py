import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/image2.jpeg')

# box filter
box_filtered = cv2.boxFilter(img, -1, (13, 13))
# gausian filter
gaussian_filtered = cv2.GaussianBlur(img, (13, 13), 0)

# plot
fig, ax = plt.subplots(3, 1, figsize=(15, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[1].imshow(cv2.cvtColor(box_filtered, cv2.COLOR_BGR2RGB))
ax[1].set_title('Box Filtered Image')
ax[2].imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
ax[2].set_title('Gaussian Filtered Image')
fig.subplots_adjust(hspace=1.0)
plt.show()
