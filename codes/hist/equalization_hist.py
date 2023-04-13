import cv2
import matplotlib.pyplot as plt

# Load image in grayscale mode
img = cv2.imread('./images/sunset.jpg', cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_img = cv2.equalizeHist(img)

# Plot original and equalized image side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(equalized_img, cmap='gray')
axes[1].set_title('Equalized')
plt.show()