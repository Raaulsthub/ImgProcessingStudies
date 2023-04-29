import cv2
import matplotlib.pyplot as plt


img = cv2.imread('./images/image2.jpeg')

# diferential 11x11 filter
kernel_size = 11
kernel = cv2.getGaussianKernel(kernel_size, 0)
kernel = kernel * kernel.T
filtered_img = cv2.filter2D(img, -1, kernel)

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title('Imagem original')
ax[1].imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
ax[1].set_title('Imagem filtrada com kernel de 11x11')
plt.show()

