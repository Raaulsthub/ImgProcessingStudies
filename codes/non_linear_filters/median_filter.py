import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/giraffe.jpg', cv2.IMREAD_GRAYSCALE)

# salt and pepper noise
noise = np.zeros(img.shape, np.uint8)
cv2.randu(noise, 0, 255)
black = noise < 10
white = noise > 245
img[black] = 0
img[white] = 255

# median filter
filtered = cv2.medianBlur(img, 5)

# Plot images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(img, cmap='gray', interpolation='nearest')
ax1.set_title('Salt and Pepper Noise')
ax1.set_xticks([])
ax1.set_yticks([])

ax2.imshow(filtered, cmap='gray', interpolation='nearest')
ax2.set_title('Median Filter')
ax2.set_xticks([])
ax2.set_yticks([])

plt.show()
