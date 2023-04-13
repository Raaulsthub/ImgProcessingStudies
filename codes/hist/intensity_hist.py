import cv2
import numpy as np
import matplotlib.pyplot as plt

# load colored image
img = cv2.imread('./images/sunset.jpg')

# get the gray image based on the color one
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# calculate histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# plot
fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Colored Image')
ax[0].axis('off')
ax[1].plot(hist, color='gray')
ax[1].set_title('Intensity Histogram')
ax[1].set_xlabel('Bins')
ax[1].set_ylabel('Frequency')
plt.show()