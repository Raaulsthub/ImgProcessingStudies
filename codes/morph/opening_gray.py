import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/street.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# opening
opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

# plotting
fig, axs = plt.subplots(1, 3, figsize=(20, 20))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(gray, cmap='gray')
axs[1].set_title('Grayscale')
axs[2].imshow(opening, cmap='gray')
axs[2].set_title('Opening')
for ax in axs:
    ax.axis('off')
plt.show()
