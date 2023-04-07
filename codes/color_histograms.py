import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load color image
img = cv2.imread('./images/sunset.jpg')

# Split image into three color channels
b,g,r = cv2.split(img)

# Calculate histograms for each color channel
hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
hist_b = cv2.calcHist([b],[0],None,[256],[0,256])

# Plot color image and histograms in one figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
plt.gray()

axs[0,0].imshow(img[:,:,::-1])
axs[0,0].set_title('Color Image')

axs[0,1].plot(hist_r, color='r')
axs[0,1].set_title('Red Channel Histogram')

axs[1,0].plot(hist_g, color='g')
axs[1,0].set_title('Green Channel Histogram')

axs[1,1].plot(hist_b, color='b')
axs[1,1].set_title('Blue Channel Histogram')

plt.tight_layout()
plt.show()
