import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


class SoyProcessingPipeline:
    def __init__(self):
        self.images = []

    # filters
    
    def filter_median(self, kernel_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.medianBlur(self.images[i], kernel_size)

    def filter_gauss(self, kernel_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.GaussianBlur(self.images[i], (kernel_size, kernel_size), 0)

    def filter_mean(self, kernel_size):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.blur(self.images[i], (kernel_size, kernel_size), 0)

    
    # color conversion

    def conversion_grayscale(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2GRAY)

    def conversion_binary_threshold(self, thresh):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], thresh, 255, cv2.THRESH_BINARY)

    def conversion_binary_otsu(self):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def conversion_binary_adaptive(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def conversion_binary_triangle(self):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    def conversion_binary_yen(self):
        for i in np.arange(len(self.images)):
            _, self.images[i] = cv2.threshold(self.images[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def conversion_binary_mean_adaptive(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    def conversion_binary_gaussian_adaptive(self):
        for i in np.arange(len(self.images)):
            self.images[i] = cv2.adaptiveThreshold(self.images[i], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    
    # loading and resizing images

    def load_and_resize_images(self, directory, target_size):
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(directory, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
                    self.images.append(resized_image)



def plot_images(images):
    # Plotting the images
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def test():
    pipe = SoyProcessingPipeline()
    # loading test image
    pipe.load_and_resize_images('./images', (640, 640))
    plot_images(pipe.images)
    pipe.filter_mean(5)
    pipe.conversion_grayscale()
    pipe.conversion_binary_threshold(127)
    plot_images(pipe.images)


def main():
    test()

if __name__ == '__main__':
    main()


