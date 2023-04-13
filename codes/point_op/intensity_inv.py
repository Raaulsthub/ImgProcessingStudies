import cv2
import matplotlib.pyplot as plt
import numpy as np

# load img in gray scale
img = cv2.imread("./images/tree.jpeg", 0)

# invertion a = (aMax - a)
inverted_img = np.max(img) - img

# img show using matplotlib
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("Original Image")

ax2.imshow(cv2.cvtColor(inverted_img, cv2.COLOR_BGR2RGB))
ax2.set_title("Inverted Image")

ax3.hist(img.ravel(), 256, [0, 256], color='red')
ax3.set_title("Original Image Histogram")
ax3.set_xlim([0, 256])

ax4.hist(inverted_img.ravel(), 256, [0, 256], color='green')
ax4.set_title("Inverted Image Histogram")
ax4.set_xlim([0, 256])

plt.show()
