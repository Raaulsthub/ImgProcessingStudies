import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/cat.jpg', 0) # Load as grayscale
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# cross kernel will be used
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# skeletonizing
skel = np.zeros(img.shape, np.uint8)
done = False

while not done:
    eroded = cv2.erode(thresh, kernel)
    temp = cv2.dilate(eroded, kernel)
    temp = cv2.subtract(thresh, temp)
    skel = cv2.bitwise_or(skel, temp)
    thresh = eroded.copy()

    zeros = img.size - cv2.countNonZero(thresh)
    if zeros == img.size:
        done = True

# plotting
plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(skel, cmap='gray'),plt.title('Skeletonized')
plt.xticks([]), plt.yticks([])
plt.show()
