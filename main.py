from collections import deque
import math
from skimage import io, data
import numpy as np
from matplotlib import pyplot as plt
import time
import numpy as np
import scipy.io
from scipy.fft import fft2, ifft2
from pathlib import Path
from scipy import ndimage


# low pass filter
def low_pass(image, filt_size):
    h = filt_size // 2
    a, b = np.shape(image)
    for i in range(h, a - h):
        for j in range(h, b - h):
            filt = np.sum(image[i - h:i + h + 1, j - h:j + h + 1])
            image[i][j] = filt / (filt_size * filt_size)
    return image


# 3D View of an image surface
def surface(image):
    a, b = np.shape(image)
    maxi = max(a, b)
    x1 = np.outer(np.linspace(0, maxi, (maxi + 1)), np.ones((maxi + 1)))
    y1 = x1.copy().T  # transpose
    x1 = x1[:a, :b]
    y1 = y1[:a, :b]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, y1, image, alpha=0.7, cmap='coolwarm')
    plt.show()


# Gradient Kernel applied on image
def sobel_gradient(image):
    gradient_mag = np.zeros_like(image).astype(float)
    gradient_direction = np.zeros_like(image).astype(float)
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.transpose(sobelx)
    gradx = ndimage.convolve(image, sobelx, mode="reflect")
    grady = ndimage.convolve(image, sobely, mode="reflect")
    gradient_mag = np.sqrt(gradx ** 2 + grady ** 2)
    return gradient_mag


# Gradient kernel applied on image
def prewitt_gradient(image):
    gradient_mag = np.zeros_like(image).astype(float)
    gradient_direction = np.zeros_like(image).astype(float)
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.transpose(prewitt_x)
    gradx = ndimage.convolve(image, prewitt_x, mode="reflect")
    grady = ndimage.convolve(image, prewitt_y, mode="reflect")
    gradient_mag = np.sqrt(gradx ** 2 + grady ** 2)
    return gradient_mag


# Gets grad map as per the required kernel after gaussian kernel
def grad_map(image, variance, threshold, operator):
    image = image.astype(int)
    filt_size = 5
    h = filt_size // 2
    spatial_gauss_kernel = np.zeros((filt_size, filt_size)).astype(float)
    gaussian_var = variance
    for i in range(-h, h + 1):
        for j in range(-h, h + 1):
            spatial_gauss_kernel[i + h, j + h] = math.exp(-(i ** 2 + j ** 2) / (2 * gaussian_var))
    spatial_gauss_kernel = spatial_gauss_kernel / np.sum(spatial_gauss_kernel)
    imagef = ndimage.convolve(image, spatial_gauss_kernel, mode="reflect").astype(int)
    if operator == "sobel":
        grad_mag = sobel_gradient(imagef)
    if operator == "prewitt":
        grad_mag = prewitt_gradient(imagef)
    return grad_mag


# Pad for convolution
def replication_padding(matrix: np.ndarray) -> np.ndarray:
    M, N = matrix.shape
    padded_matrix = np.zeros((M + 2, N + 2))
    padded_matrix[1:M + 1, 1:N + 1] = matrix
    pad_indicator_matrix = np.zeros_like(padded_matrix)
    for i in range(0, M + 2):
        if i != 0 and i != M + 1:
            padded_matrix[i, 0] = padded_matrix[i, 1]
            pad_indicator_matrix[i, 0] = 1
            padded_matrix[i, N + 1] = padded_matrix[i, N]
            pad_indicator_matrix[i, N + 1] = 1
        elif i == 0:
            padded_matrix[0, 0] = padded_matrix[1, 1]
            pad_indicator_matrix[0, 0] = 1
            padded_matrix[0, N + 1] = padded_matrix[1, N]
            pad_indicator_matrix[0, N + 1] = 1
            for j in range(1, N + 1):
                padded_matrix[0, j] = padded_matrix[1, j]
                pad_indicator_matrix[0, j] = 1
        else:
            padded_matrix[M + 1, 0] = padded_matrix[M, 1]
            pad_indicator_matrix[M + 1, 0] = 1
            padded_matrix[M + 1, N + 1] = padded_matrix[M, N]
            pad_indicator_matrix[M + 1, N + 1] = 1
            for j in range(1, N + 1):
                padded_matrix[M + 1, j] = padded_matrix[M, j]
                pad_indicator_matrix[M + 1, j] = 1
    return padded_matrix, pad_indicator_matrix


# watershed algorithm
def watershed_algorithm(image, R):
    Mask = -2
    Wshed = 0
    INIT = -1
    Undef = -5
    distance_matrix = np.zeros_like(image)
    label_matrix = np.ones_like(image) * INIT
    current_label = 0
    FIFO = deque([])
    pixel_intensities = np.unique(image)
    M, N = np.shape(image)

    for i in pixel_intensities:
        label_matrix = np.where(image == i, Mask, label_matrix)
        label_matrix = np.where(R == 1, Undef, label_matrix)
        ri, ci = np.nonzero(image == i)
        for j, k in zip(ri, ci):
            if R[j, k] == 1:
                continue
            Nbd = label_matrix[j - 1:j + 2, k - 1:k + 2]
            unique_labels = np.flatnonzero(Nbd > 0)
            watershed = np.flatnonzero(Nbd == Wshed)
            if len(unique_labels) != 0 or len(watershed) != 0:
                distance_matrix[j, k] = 1
                FIFO.appendleft((j, k))
        current_distance = 1
        FIFO.appendleft("fict")
        while (True):
            popped = FIFO.pop()
            if popped == "fict":
                if len(FIFO) == 0:
                    break
                else:
                    FIFO.appendleft("fict")
                    current_distance += 1
                    popped = FIFO.pop()
            idx1, idx2 = popped
            for j in range(idx1 - 1, idx1 + 2):
                for k in range(idx2 - 1, idx2 + 2):
                    if j == idx1 and k == idx2:
                        continue
                    if distance_matrix[j, k] < current_distance and (
                            label_matrix[j, k] > 0 or label_matrix[j, k] == Wshed):
                        if label_matrix[j, k] > 0:
                            if label_matrix[idx1, idx2] == Mask or label_matrix[idx1, idx2] == Wshed:
                                label_matrix[idx1, idx2] = label_matrix[j, k]
                            elif label_matrix[idx1, idx2] != label_matrix[j, k]:
                                label_matrix[idx1, idx2] = Wshed
                        elif label_matrix[idx1, idx2] == Mask:
                            label_matrix[idx1, idx2] = Wshed
                    elif distance_matrix[j, k] == 0 and label_matrix[j, k] == Mask:
                        distance_matrix[j, k] = current_distance + 1
                        FIFO.appendleft((j, k))
        for j, k in zip(ri, ci):
            if R[j, k] == 1:
                continue
            distance_matrix[j, k] = 0
            if label_matrix[j, k] == Mask:
                current_label += 1
                FIFO.appendleft((j, k))
                label_matrix[j, k] = current_label
                while (FIFO):
                    idx1, idx2 = FIFO.pop()
                    for l in range(idx1 - 1, idx1 + 2):
                        for m in range(idx2 - 1, idx2 + 2):
                            if l == idx1 and m == idx2:
                                continue
                            if label_matrix[l, m] == Mask:
                                FIFO.appendleft((l, m))
                                label_matrix[l, m] = current_label

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            ls = label_matrix[i - 1:i + 2, j - 1:j + 2]
            X = np.sort(ls, axis=None)
            X = X[X != -5]
            X = X[X != 0]
            if len(X):
                if X[0] < label_matrix[i, j]:
                    label_matrix[i, j] = Wshed
    return label_matrix


if __name__ == '__main__':
    image_path = "./Data/images/water_coins.jpg"
    image_color = io.imread(image_path)
    image = io.imread(image_path, as_gray=True)
    image = image * 255
    # =============================================================================
    #     if single channel image
    #     image_color=np.dstack((image,image,image))
    # =============================================================================
    image_filtered = low_pass(image, 3)
    image_filtered = low_pass(image_filtered, 3)
    image_filtered = low_pass(image_filtered, 3)
    grad = grad_map(image_filtered, variance=10, threshold=50, operator="sobel")
    input_image = (grad / (np.max(grad))) * 10
    # surface(grad)
    input_image = input_image.astype('uint8')
    image, R = replication_padding(input_image)
    M, N = np.shape(image)
    label_matrix = watershed_algorithm(image, R)
    image_color[label_matrix[1:M - 1, 1:N - 1] == 0] = [255, 0, 0]
    plt.imshow(image_color)
    plt.show()













