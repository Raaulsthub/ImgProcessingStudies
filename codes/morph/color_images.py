import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/cat.jpg')

# splitting the BGR image into three channels
b, g, r = cv2.split(img)

# kernel
kernel = np.ones((15,15),np.uint8)

# opening on each color channel
r_open = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel)
g_open = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
b_open = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)

# merging channels on image
opened_img = cv2.merge((r_open, g_open, b_open))

# plotting
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opened_img),plt.title('Opened')
plt.xticks([]), plt.yticks([])
plt.show()
