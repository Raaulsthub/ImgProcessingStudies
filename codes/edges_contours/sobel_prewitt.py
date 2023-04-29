import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/rhino.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_eq = cv2.equalizeHist(gray)

# apply the sobel operator
sobelx = cv2.Sobel(gray_eq, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_eq, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(np.square(sobelx) + np.square(sobely))

# apply the prewitt operator
prewittx = cv2.filter2D(gray_eq, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
prewitty = cv2.filter2D(gray_eq, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
prewitt = np.sqrt(np.square(prewittx) + np.square(prewitty))

# plot
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(2, 2, 2)
plt.imshow(gray_eq, cmap='gray')
plt.title('Grayscale Image (Equalized)')
plt.subplot(2, 2, 3)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Operator')
plt.subplot(2, 2, 4)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt Operator')
plt.show()

