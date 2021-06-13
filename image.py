import numpy as np
import png


class Image:
    def __init__(self, x_pixels=0, y_pixels=0, num_channels=0, filename=''):
        # either filename, or all the other values are required.
        self.input_path = 'input/'
        self.output_path = 'output/'

        if x_pixels and y_pixels and num_channels:
            self.x_pixels = x_pixels
            self.y_pixels = y_pixels
            self.num_channels = num_channels
            self.array = np.zeros((x_pixels, y_pixels, num_channels))
        elif filename:
            self.array = self.read_image(filename)
            self.x_pixels, self.y_pixels, self.num_channels = self.array.shape
        else:
            raise ValueError(
                'Please specify either a filename OR specify the dimensions of the image')

    def read_image(self, filename, gamma=2.2):
        """read PNG RGB image, return 3D numpy array organized along Y, X, channel
        values are float, gamma is decoded

        Args:
            filename (String): image file name in the input directory
            gamma (float, optional): factor to resize the input image.
            Defaults to 2.2.

        Returns:
            resized_image (array): numpy array with pixel values of the image.
        """

        image = png.Reader(self.input_path + filename).asFloat()
        resized_image = np.vstack(list(image[2]))
        resized_image.resize(image[1], image[0], 3)
        resized_image = resized_image ** gamma

        return resized_image

    def write_image(self, output_filename, gamma=2.2):
        """3D numpy array (Y, X, channel) of values between 0 and 1 -> write to png

        Args:
            output_filename (String): give a name for the file to be stored.
            gamma (float, optional): factor to resize the input image.
            Defaults to 2.2.
        """

        image = np.clip(self.array, 0, 1)
        y, x = self.array.shape[0], self.array.shape[1]
        image = image.reshape(y, x*3)
        writer = png.Writer(x, y)

        with open(self.output_path + output_filename, 'wb') as handle:
            writer.write(handle, 255 * (image ** (1/gamma)))
        self.array.resize(y, x, 3)


if __name__ == '__main__':
    image = Image(filename='lake.png')
    image.write_image('test.png')
