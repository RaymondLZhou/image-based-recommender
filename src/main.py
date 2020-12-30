import images
import transfer
from constants import *

if __name__ == '__main__':
    content_image = images.load_image(content_path)
    style_image = images.load_image(style_path)
    inputs = {'content_image': content_image, 'style_image': style_image}

    images.plot_images(content_image, style_image)

    image = tf.Variable(content_image)
    transfer.transfer_style(image, epochs, steps_per_epoch, optimizer, layers, inputs, weights)
    images.convert_tensor_to_image(image).save(output_path)

    new_image = images.load_image(output_path)
    images.plot_images(content_image, style_image, new_image)
