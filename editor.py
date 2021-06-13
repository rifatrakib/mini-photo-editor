from image import Image
import numpy as np


def brighten(image, factor):
    """when we brighten, we just want to make each channel higher by some amount.
    factor is a value > 0, how much you want to brighten the image by (< 1 = darken, > 1 = brighten)

    Args:
        image (Image): an instance of Image class from image.py
        factor (float): value referring to the brightness of the image
    """
    # x, y -> pixel of the image (x * y), num_channels -> RGB
    x_pixels, y_pixels, num_channels = image.array.shape
    # create a new blank image
    new_image = Image(
        x_pixels=x_pixels, y_pixels=y_pixels,
        num_channels=num_channels,
    )
    # change brightness using numpy vector
    new_image.array = image.array * factor
    return new_image


def contrast(image, factor, mid):
    """adjust the contrast by increasing the difference from the user-defined midpoint by factor amount

    Args:
        image (Image): an instance of Image class from image.py
        factor (float): value referring to the brightness of the image
        default (float): equilibrium value/normal contrast/No contrast
    """
    x_pixels, y_pixels, num_channels = image.array.shape  # x, y -> pixel of the image (x * y), num_channels -> RGB
    # create a new blank image
    new_image = Image(
        x_pixels=x_pixels, y_pixels=y_pixels,
        num_channels=num_channels,
    )
    # change contrast using numpy vector
    new_image.array = (image.array - mid) * factor + mid
    return new_image


def blur(image, factor):
    """factor is the number of pixels to take into account when applying the blur,
    i.e., factor = 3 would be neighbors to the left/right, top/bottom, and diagonals.
    For simplicity, factor should always be an *odd* number.

    Args:
        image (Image): an instance of Image class from image.py
        factor (float)): float number indicating the scope of blurriness
    """
    # x, y -> pixel of the image (x * y), num_channels -> RGB
    x_pixels, y_pixels, num_channels = image.array.shape
    # create a new blank image
    new_image = Image(
        x_pixels=x_pixels, y_pixels=y_pixels,
        num_channels=num_channels,
    )
    # this is a variable that tells us how many neighbors we actually look at (ie for a kernel of 3, this value should be 1)
    radius = factor // 2
    for x_px in range(x_pixels):
        for y_px in range(y_pixels):
            for ch in range(num_channels):
                # using a naive implementation of iterating through each neighbor and summing
                total = 0
                for x in range(max(0, x_px-radius), min(new_image.x_pixels-1, x_px+radius) + 1):
                    for y in range(max(0, y_px-radius), min(new_image.y_pixels-1, y_px+radius) + 1):
                        total += image.array[x, y, ch]
                new_image.array[x_px, y_px, ch] = total / (radius ** 2)
    return new_image


def vectorize(image, factor):
    """the factor should be a 2D array that represents the matrix we'll use!
    For simiplicity of this implementation, let's assume that the kernel is SQUARE.
    For example the sobel x kernel (detecting horizontal edges) is as follows:
    [1 0 -1]
    [2 0 -2]
    [1 0 -1]

    Args:
        image (Image): an instance of Image class from image.py
        factor (float)): float number indicating the scope of blurriness
    """
    # x, y -> pixel of the image (x * y), num_channels -> RGB
    x_pixels, y_pixels, num_channels = image.array.shape
    # create a new blank image
    new_image = Image(
        x_pixels=x_pixels, y_pixels=y_pixels,
        num_channels=num_channels,
    )
    # this is a variable that tells us how many neighbors we actually look at (ie for a kernel of 3, this value should be 1)
    radius = factor.shape[0] // 2
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0, x-radius), min(new_image.x_pixels-1, x+radius)+1):
                    for y_i in range(max(0, y-radius), min(new_image.y_pixels-1, y+radius)+1):
                        x_k = x_i + radius - x
                        y_k = y_i + radius - y
                        kernel_val = factor[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_image.array[x, y, c] = total
    return new_image


def merge_images(image1, image2):
    """combine two images using the squared sum of squares:
    value = sqrt(value_1**2, value_2**2)
    size of image1 and image2 MUST be the same

    Args:
        image1 (Image): an instance of Image class from image.py
        image2 (Image): an instance of Image class from image.py
    """
    # x, y -> pixel of the image (x * y), num_channels -> RGB
    x_pixels, y_pixels, num_channels = image1.array.shape
    # create a new blank image
    new_image = Image(
        x_pixels=x_pixels, y_pixels=y_pixels,
        num_channels=num_channels,
    )
    for x_px in range(x_pixels):
        for y_px in range(y_pixels):
            for ch in range(num_channels):
                new_image.array[x_px, y_px, ch] = (
                    image1.array[x_px, y_px, ch]**2 + image2.array[x_px, y_px, ch]**2) ** 0.5
    return new_image


if __name__ == '__main__':
    lake = Image(filename='lake.png')
    city = Image(filename='city.png')

    # brightening
    brightened = brighten(lake, 1.7)
    brightened.write_image('increase-brighteness.png')

    # darkening
    darkened_im = brighten(lake, 0.3)
    darkened_im.write_image('decrease-brightness.png')

    # increase contrast
    incr_contrast = contrast(lake, 2, 0.5)
    incr_contrast.write_image('increased-contrast.png')

    # decrease contrast
    decr_contrast = contrast(lake, 0.5, 0.5)
    decr_contrast.write_image('decreased-contrast.png')

    # blur using kernel 3
    blur_3 = blur(city, 3)
    blur_3.write_image('blur-r3.png')

    # blur using kernel size of 15
    blur_15 = blur(city, 15)
    blur_15.write_image('blur-r15.png')

    # let's apply a sobel edge detection kernel on the x and y axis
    sobel_x = vectorize(city, np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    sobel_x.write_image('edge-x.png')
    sobel_y = vectorize(city, np.array(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_y.write_image('edge-y.png')

    # let's combine these and make an edge detector!
    sobel_xy = merge_images(sobel_x, sobel_y)
    sobel_xy.write_image('edge-xy.png')
