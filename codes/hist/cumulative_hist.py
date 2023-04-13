import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/sunset.jpg', cv2.IMREAD_GRAYSCALE)

# normal hist
hist, bins = np.histogram(img.ravel(), 256, [0, 256])

# cumulative hist
cumulative_hist = np.cumsum(hist)

# ploting
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(img, cmap='gray')
ax[0].set_title('Grayscale Image')

ax[1].hist(img.ravel(), 256, [0, 256])
ax[1].set_xlim([0, 256])
ax[1].set_title('Histogram')

ax[2].plot(cumulative_hist)
ax[2].set_xlim([0, 256])
ax[2].set_title('Cumulative Histogram')

plt.show()