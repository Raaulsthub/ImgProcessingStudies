import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('./images/image.jpeg')

# resize (image used was too big)
new_width = int(img.shape[1] * 0.2)
new_height = int(img.shape[0] * 0.2)
resized_img = cv2.resize(img, (new_width, new_height))

# create list of kernel sizes
kernel_sizes = [5, 15, 21, 25]

# create a 2x2 subplot figure with blue background
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10), facecolor='skyblue')

# iterate through kernel sizes and apply median blur to each
for i, size in enumerate(kernel_sizes):
    row = i // 2
    col = i % 2
    kernel = np.ones((size, size), np.float32) / (size * size)
    blurred_img = cv2.filter2D(resized_img, -1, kernel)
    axs[row][col].imshow(blurred_img[:,:,::-1])
    axs[row][col].set_title(f'{size}x{size} Kernel')
    axs[row][col].set_xticks([]), axs[row][col].set_yticks([])

# show the plot
plt.show()