import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.applications import vgg16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

nb_closest_images = 5
img_path = "../data/images/"

vgg_model = vgg16.VGG16(weights='imagenet')
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
feat_extractor.summary()

image_width = eval(str(vgg_model.layers[0].output.shape[1]))
image_height = eval(str(vgg_model.layers[0].output.shape[2]))

pil_img = load_img('../data/images/10000.jpg',  target_size=(image_width, image_height))
array_img = img_to_array(pil_img)
images = np.expand_dims(array_img, axis=0)

print(images.shape)
