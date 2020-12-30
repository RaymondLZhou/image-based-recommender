from hyperparameters import *

image_path = '../data/'
content_path = image_path + 'content/' + content_name + '.jpg'
style_path = image_path + 'style/' + style_name + '.jpg'
output_path = image_path + 'output/' + content_name + style_name + '.jpg'

content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
layers = {'content_layers': content_layers, 'style_layers': style_layers}
