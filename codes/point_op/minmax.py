import cv2
import matplotlib.pyplot as plt

# load the images
img1 = cv2.imread('./images/tree.jpeg')
img2 = cv2.imread('./images/deer.jpeg')

# check if images have same size
if img1.shape != img2.shape:
    # resize img2 to the size of img1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# aplly max value and min value ops
min_img = cv2.min(img1, img2)
max_img = cv2.max(img1, img2)

# plot
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
axes[0, 0].imshow(img1)
axes[0, 0].set_title('Image 1')
axes[0, 1].imshow(img2)
axes[0, 1].set_title('Image 2')
axes[1, 0].imshow(cv2.cvtColor(min_img, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('cv2.min()')
axes[1, 1].imshow(cv2.cvtColor(max_img, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('cv2.max()')
plt.show()