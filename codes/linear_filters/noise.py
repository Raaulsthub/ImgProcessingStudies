import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/image2.jpeg')

# conver to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gausian noise
gaussian_noise = np.zeros_like(gray)
cv2.randn(gaussian_noise, 0, 50)
noisy_image = gray + gaussian_noise

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.patch.set_facecolor('xkcd:grass green')
ax[0].imshow(gray, cmap='gray')
ax[0].set_title('Grayscale Image')
ax[1].imshow(noisy_image, cmap='gray')
ax[1].set_title('Gaussian Noise Image')
plt.show()

