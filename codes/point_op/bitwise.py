import cv2
import matplotlib.pyplot as plt
import numpy as np

# load the images
img1 = cv2.imread('./images/tree.jpeg')
img2 = cv2.imread('./images/deer.jpeg')

# check if images have same size
if img1.shape != img2.shape:
    # resize img2 to the size of img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))


# gray scale conversion
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# otsu threshold
_, binary1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, binary2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# bitwise ops
and_img = cv2.bitwise_and(binary1, binary2)
nand_img = cv2.bitwise_not(and_img)
or_img = cv2.bitwise_or(binary1, binary2)
nor_img = cv2.bitwise_not(or_img)

#plot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
axes[0, 0].imshow(img1, cmap='gray')
axes[0, 0].set_title('Image 1')
axes[0, 1].imshow(binary1, cmap='gray')
axes[0, 1].set_title('Binary 1')
axes[0, 2].imshow(cv2.cvtColor(and_img, cv2.COLOR_GRAY2RGB))
axes[0, 2].set_title('AND')
axes[1, 0].imshow(cv2.cvtColor(nand_img, cv2.COLOR_GRAY2RGB))
axes[1, 0].set_title('NAND')
axes[1, 1].imshow(cv2.cvtColor(or_img, cv2.COLOR_GRAY2RGB))
axes[1, 1].set_title('OR')
axes[1, 2].imshow(cv2.cvtColor(nor_img, cv2.COLOR_GRAY2RGB))
axes[1, 2].set_title('NOR')
for ax in axes.flatten():
    ax.axis('off')
plt.show()
