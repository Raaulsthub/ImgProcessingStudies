import cv2
import matplotlib.pyplot as plt


img = cv2.imread('./images/image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel size defines size of the objects being recovered
kernel_size = 25
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

# black top hat operation
black_tophat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# white top hat operation
white_tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# plotting images
fig, axs = plt.subplots(1, 3, figsize=(10, 10))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original')
axs[1].imshow(black_tophat, cmap='gray')
axs[1].set_title('Black Top Hat')
axs[2].imshow(white_tophat, cmap='gray')
axs[2].set_title('White Top Hat')
plt.show()