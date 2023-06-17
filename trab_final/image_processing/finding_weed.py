import numpy as np
from matplotlib import pyplot as plt
import cv2
from processing_pipeline import ImageProcessingPipeline


def find_weed_point_density(images):

    scale = 0.2

    pipeline = ImageProcessingPipeline()


    # pipeline.load_images('./images/')
    pipeline.images = images.copy()
    
    images = pipeline.images.copy()

    pipeline.filter_gauss(5)

    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    pipeline.color_mask_hsv(lower_green, upper_green)


    pipeline.morph_opening(ksize=5)
    pipeline.morph_erosion(ksize=5)


    pipeline.conversion_grayscale()
    pipeline.conversion_binary_otsu()
    pipeline.morph_opening()
    pipeline.edge_detection_canny()


    # density
    # Define the block size
    block_size = 150
    # Define the threshold for white pixels in the block
    threshold = 1000
    for idx in range(len(pipeline.images)):
        # Define a list to keep track of the blocks with more white pixels than the threshold
        blocks = []
        # Iterate through the binary image to find blocks with more white pixels than the threshold
        height, width = pipeline.images[idx].shape
        for i in range(0, height-block_size, block_size):
            for j in range(0, width-block_size, block_size):
                block = pipeline.images[idx][i:i+block_size, j:j+block_size]
                if cv2.countNonZero(block) > threshold:
                    # Check if the block has at least one neighbor that also has more white pixels than the threshold
                    has_neighbor = False
                    for x in range(max(i-block_size, 0), min(i+2*block_size, height-block_size), block_size):
                        for y in range(max(j-block_size, 0), min(j+2*block_size, width-block_size), block_size):
                            if x == i and y == j:
                                continue
                            neighbor = pipeline.images[idx][x:x+block_size, y:y+block_size]
                            if cv2.countNonZero(neighbor) > threshold:
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break
                    
                    # If the block has at least one neighbor, add it to the list of blocks to plot
                    if has_neighbor:
                        blocks.append((i, j))
                        print(f"Block found at ({i}, {j}) with {cv2.countNonZero(block)} white pixels")

        # Plot the red rectangles only for the blocks with at least one neighbor
        for block in blocks:
            i, j = block
            cv2.rectangle(images[idx], (j, i), (j+block_size, i+block_size), (0, 0, 255), 10)

    pipeline.images = images

    pipeline.resize_by_size((640, 640))

    # pipeline.plot_images()
    return pipeline.images


def alternative():
    pipeline = ImageProcessingPipeline()

    # loading
    pipeline.load_and_resize_images('./images/', (640, 640))
    # pipeline.plot_images()

    # filter
    pipeline.filter_median(3)
    # pipeline.plot_images()

    # green colors only
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    pipeline.color_mask_hsv(lower_green, upper_green)
    # pipeline.plot_images()

    # convert to grayscale
    pipeline.conversion_grayscale()
    pipeline.plot_images()

    # morph open
    pipeline.morph_dilation()
    pipeline.plot_images()

    # canny edge detection
    pipeline.edge_detection_canny()
    pipeline.plot_images()



def find_weed_on_video():
    cap = cv2.VideoCapture("./videos/video2.MP4")

    scale = 0.2


    # Loop through the frames of the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        images = [frame.copy()]
        images = find_weed_point_density(images)


        # Show the original image and the mask with edges and contours in different windows
        frame = cv2.resize(frame, (640, 640))

        cv2.imshow("Original", frame)
        cv2.imshow("Detection", images[0])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    




if __name__ == '__main__':
    find_weed_on_video()