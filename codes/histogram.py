import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('./images/sunset.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate histogram
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

# Plot the histogram
plt.subplot(121)
plt.hist(img.flatten(), 256, [0, 256], color='gray')
plt.xlim([0, 256])
plt.xlabel('Tones')
plt.ylabel('Pixel count')

# Display the image
plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

# Show the plot
plt.show()