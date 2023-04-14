import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('./images/tree.jpeg')

# gamma value definition
gamma = 1.5

# create table for gamma operation
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# apply gamma to the image
img_corrigida = cv2.LUT(img, table)

# plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(img_corrigida, cv2.COLOR_BGR2RGB))
ax[0].set_title('Imagem Original')
ax[1].set_title('Imagem Corrigida')
ax[0].axis('off')
ax[1].axis('off')
plt.show()