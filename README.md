# Image Segmentation - Watershed Algorithm
This project is part of a course from **Digital Image Processing (DIP)** at IISc.

# Introduction
The watershed algorithm is a technique used for image segmentation, which divides an image into multiple regions based on similarities between pixels. The algorithm is based on the concept of a watershed, which is a geographical feature that separates two areas of land by defining the boundary between the two basins.

In the context of image processing, the watershed algorithm works by treating the image as a topographical map, where bright pixels correspond to high points and dark pixels correspond to low points. The algorithm then floods the image from its local minima, simulating the process of water flowing into the basins.

As the water flows, it gradually fills up the basins until the water from different basins meets at a ridge line. At this point, the algorithm creates a watershed boundary at the ridge line, separating the two basins. The process is repeated for all the local minima in the image, resulting in a segmentation of the image into multiple regions.


https://user-images.githubusercontent.com/81372735/224679091-5c63040f-62ff-4c43-94ac-760bec458406.mp4


The watershed algorithm has been used in a wide range of applications, including medical imaging, remote sensing, and industrial inspection. However, the algorithm is sensitive to noise and can result in over-segmentation if not used carefully. We have used preprocessing techniques like Gaussian Smoothing, Low pass filter, quantization of pixel values to mitigate the same.
# Flow Chart of Raw Watershed Algorithm
Brief of watershed algorithm as implemented in main.py

<img width="562" alt="FC_1" src="https://user-images.githubusercontent.com/81372735/224673893-fbaefb61-eeda-4fb6-9c94-33de76bafc1f.PNG">
<img width="562" alt="FC_2" src="https://user-images.githubusercontent.com/81372735/224674067-3dc6a49d-7dcc-410b-8173-dc6190e4c114.PNG">


# Over-Segmentation Issue
Raw Watershed algorithm is prone to over-segmentation issues given the image is highly likely to be noisy. 
<img width="551" alt="Over-Segmentation Issue-1" src="https://user-images.githubusercontent.com/81372735/224668913-6e487665-52e0-448b-9527-3c6aa0cfdc32.PNG">

# After Pre-Processing 
We need to smoothen out the image
using image procesing techniques. The results obtained are much better.
<img width="373" alt="after-preprocessing" src="https://user-images.githubusercontent.com/81372735/224674198-caf1db54-9eaf-4652-a884-5630a2c1da6c.PNG">

# Set Up

- pip install -r requirements.txt
- python main.py
