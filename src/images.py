import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

def load_image(image_path):
    max_dim = 512

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image

def show_image(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)

    if title:
        plt.title(title)

def plot_images(content_image, style_image, new_image=[]):
    cols = 3 if new_image != [] else 2

    plt.figure(figsize=(15, 15)) 

    plt.subplot(1, cols, 1)
    show_image(content_image, 'Content Image')

    plt.subplot(1, cols, 2)
    show_image(style_image, 'Style Image')

    if cols == 3:
        plt.subplot(1, cols, 3)
        show_image(new_image, 'New Image')

    plt.show()

def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    
    return PIL.Image.fromarray(tensor)
