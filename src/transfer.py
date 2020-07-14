import tensorflow as tf

import images
import model
import learn

content_name = 'dog'
style_name = 'NASA'

epochs = 10
steps_per_epoch = 20

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30
weights = {'style_weight': style_weight, 'content_weight': content_weight, 'total_variation_weight': total_variation_weight}

image_path = '../data/'


content_layers = ['block5_conv2'] 
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
layers = {'content_layers': content_layers, 'style_layers': style_layers}

content_path = image_path + 'content/' + content_name + '.jpg'
style_path = image_path + 'style/' + style_name + '.jpg'
output_path = image_path + 'output/' + content_name + style_name + '.jpg'

content_image = images.load_image(content_path)
style_image = images.load_image(style_path)
inputs = {'content_image': content_image, 'style_image': style_image}


images.plot_images(content_image, style_image)

image = tf.Variable(content_image)
learn.transfer_style(image, epochs, steps_per_epoch, optimizer, layers, inputs, weights)
images.tensor_to_image(image).save(output_path)

new_image = images.load_image(output_path)
images.plot_images(content_image, style_image, new_image)
