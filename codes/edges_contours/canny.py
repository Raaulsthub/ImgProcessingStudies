import cv2
import matplotlib.pyplot as plt


image = cv2.imread('./images/rhino.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# canny edge detection
edges = cv2.Canny(gray, 100, 200)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Image')
ax1.axis('off')
ax2.imshow(edges, cmap='gray')
ax2.set_title('Canny Edges')
ax2.axis('off')
plt.show()