# Image Segmentation - Watershed Algorithm
This project is part of a course from **Digital Image Processing (DIP)** at IISc.

# Introduction
The watershed algorithm is a technique used for image segmentation, which divides an image into multiple regions based on similarities between pixels. The algorithm is based on the concept of a watershed, which is a geographical feature that separates two areas of land by defining the boundary between the two basins.

In the context of image processing, the watershed algorithm works by treating the image as a topographical map, where bright pixels correspond to high points and dark pixels correspond to low points. The algorithm then floods the image from its local minima, simulating the process of water flowing into the basins.

As the water flows, it gradually fills up the basins until the water from different basins meets at a ridge line. At this point, the algorithm creates a watershed boundary at the ridge line, separating the two basins. The process is repeated for all the local minima in the image, resulting in a segmentation of the image into multiple regions.

The watershed algorithm has been used in a wide range of applications, including medical imaging, remote sensing, and industrial inspection. However, the algorithm is sensitive to noise and can result in over-segmentation if not used carefully. We have used preprocessing techniques like Gaussian Smoothing, Low pass filter, quantization of pixel values to mitigate the same.
# Flow Chart of Raw Watershed Algorithm

# Over-Segmentation Issue

# After Pre-Processing 

# Set Up

- pip install -r requirements.txt
- python main.py