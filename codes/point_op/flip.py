import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./images/tree.jpeg')

# flip image vertically and horizontally
flipped_h = cv2.flip(img, 0)
flipped_v = cv2.flip(img, 1)

# plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original')
axes[1].imshow(cv2.cvtColor(flipped_h, cv2.COLOR_BGR2RGB))
axes[1].set_title('Horizontally Flipped')
axes[2].imshow(cv2.cvtColor(flipped_v, cv2.COLOR_BGR2RGB))
axes[2].set_title('Vertically Flipped')
for ax in axes:
    ax.axis('off')
plt.show()
