import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./images/image.jpeg')

# resize (image used was too big)
new_width = int(img.shape[1] * 0.2)
new_height = int(img.shape[0] * 0.2)
resized_img = cv2.resize(img, (new_width, new_height))

# median blure, 15x15 kernel
blurred_img = cv2.medianBlur(resized_img, 15)

# plot
plt.subplot(121)
plt.imshow(resized_img[:,:,::-1])
plt.title('Resized Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(blurred_img[:,:,::-1])
plt.title('Blurred Image')
plt.xticks([]), plt.yticks([])
plt.show()


