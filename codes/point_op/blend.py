import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('./images/tree.jpeg')
img2 = cv2.imread('./images/deer.jpeg')
# check if images have same size
if img1.shape != img2.shape:
    # resize img2 to the size of img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# blend the images
alpha = 0.7
blended = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)

# plot the images
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title('Image 1')
axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title('Image 2')
axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
axes[2].set_title('Blended')
for ax in axes:
    ax.axis('off')
plt.show()
