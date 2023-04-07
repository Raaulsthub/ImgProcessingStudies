import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))
img1 = cv2.imread('./images/sunset.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./images/sunset.jpg')
plt.subplot(2,2,1)
plt.imshow(img1,cmap='gray')
plt.subplot(2,2,2)
plt.grid(True)
plt.hist(img1.ravel(),64,[0,256])
plt.xlim([0,256])
plt.subplot(2,2,3)
plt.grid(True)
plt.hist(img1.ravel(),128,[0,256])
plt.xlim([0,256])
plt.subplot(2,2,4)
plt.grid(True)
plt.hist(img1.ravel(),256,[0,256])
plt.xlim([0,256])

plt.show()