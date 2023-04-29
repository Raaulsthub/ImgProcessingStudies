import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread('./images/rhino.jpg')

# defining the unsharp mask parameters
kernel_size = (5, 5)
sigma = 1.5
amount = 1.0
threshold = 0

# creating the unsharp mask kernel
kernel = np.zeros((kernel_size[0], kernel_size[1]), dtype=np.float32)
kernel[int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2)] = 2.0
kernel /= np.sum(kernel)

# applying the unsharp mask using cv2
img_blur = cv2.GaussianBlur(img, kernel_size, sigma)
img_usm = cv2.addWeighted(img, 1 + amount, img_blur, -amount, 0)

# plot
fig, ax = plt.subplots(1, 2)
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original')
ax[1].imshow(cv2.cvtColor(img_usm, cv2.COLOR_BGR2RGB))
ax[1].set_title('Unsharp masked')
plt.show()
