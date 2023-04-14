import cv2
import matplotlib.pyplot as plt


img = cv2.imread('./images/deer.jpeg', cv2.IMREAD_GRAYSCALE)

# normalization
img_stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# plot both images and histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Image Stretching Example', fontsize=16)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[1, 0].hist(img.ravel(), 256, [0, 256], color='gray')
axs[1, 0].set_title('Original Histogram')
axs[0, 1].imshow(img_stretched, cmap='gray')
axs[0, 1].set_title('Stretched Image')
axs[1, 1].hist(img_stretched.ravel(), 256, [0, 256], color='gray')
axs[1, 1].set_title('Stretched Histogram')
plt.show()
