import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./images/street.jpeg', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# opening
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(thresh, cmap='gray')
axs[0].set_title('Binary Image')
axs[1].imshow(opened, cmap='gray')
axs[1].set_title('Opened Image')
plt.show()
