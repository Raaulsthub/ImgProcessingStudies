import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./images/tree.jpeg', 0)

# apply otsu
otsu_thresh, otsu_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# apply binary
binary_thresh, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# display image, otsu and binary
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original Image')
ax[1].imshow(otsu_img, cmap='gray')
ax[1].set_title('Otsu Thresholding')
ax[2].imshow(binary_img, cmap='gray')
ax[2].set_title('Binary Thresholding')
plt.show()
