import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('./images/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply salt and pepper noise
noise = np.zeros_like(gray)
cv2.randu(noise, 0, 255)
salt = noise > 245
pepper = noise < 10
gray[salt] = 255
gray[pepper] = 0

# Apply median filter with custom kernel
kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], dtype=np.uint8)
filtered = cv2.medianBlur(gray, 5)

# Plot the noisy and filtered images using matplotlib
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(gray, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(filtered, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')
plt.show()
