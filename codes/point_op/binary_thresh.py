import cv2
import matplotlib.pyplot as plt

# load image in grayscale
img = cv2.imread('./images/tree.jpeg', cv2.IMREAD_GRAYSCALE)

# apply binary thresholding with threshold value of 127
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# plot original and thresholded images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(thresh, cmap='gray')
axs[1].set_title('Thresholded Image')
axs[1].axis('off')
plt.show()
