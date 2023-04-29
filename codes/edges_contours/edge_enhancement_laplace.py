import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/rhino.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Laplacian operator for edge enhancement
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
# Rescale values to lie between 0 and 255 and convert to uint8
laplacian = np.uint8(np.absolute(laplacian))
enhanced = cv2.addWeighted(gray, 1.5, laplacian, -0.5, 0)


fig, axs = plt.subplots(1, 2)
axs[0].imshow(gray, cmap='gray')
axs[0].set_title('Gray')
axs[0].axis('off')
axs[1].imshow(enhanced, cmap='gray')
axs[1].set_title('Enhanced')
axs[1].axis('off')
plt.show()

