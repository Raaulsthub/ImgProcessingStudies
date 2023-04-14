import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('./images/tree.jpeg')
img2 = cv2.imread('./images/deer.jpeg')

# resize images if needed
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# add images
img_adicionada = cv2.add(img1, img2)

# plot
img_concat = np.concatenate((img1, img2, img_adicionada), axis=1)
plt.imshow(cv2.cvtColor(img_concat, cv2.COLOR_BGR2RGB))
plt.show()