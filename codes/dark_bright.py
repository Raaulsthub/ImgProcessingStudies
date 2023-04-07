import cv2
from matplotlib import pyplot as plt

# load bright and dark images
img1 = cv2.imread('./images/dark.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/bright.jpeg', cv2.IMREAD_GRAYSCALE)

# create a figure for the subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# show first image
axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title('Image 1')

# show histogram of first image
axs[1, 0].hist(img1.ravel(), 256, [0, 256])
axs[1, 0].set_title('Histogram of Image 1')

# show second image
axs[0, 1].imshow(img2, cmap='gray')
axs[0, 1].set_title('Image 2')

# show histogram of second image
axs[1, 1].hist(img2.ravel(), 256, [0, 256])
axs[1, 1].set_title('Histogram of Image 2')

# space ajusts
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)

plt.show()