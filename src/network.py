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
imgs_path = "../data/images/"

vgg_model = vgg16.VGG16(weights='imagenet')
feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)
feat_extractor.summary()

image_width = eval(str(vgg_model.layers[0].output.shape[1]))
image_height = eval(str(vgg_model.layers[0].output.shape[2]))

files = [imgs_path + x for x in os.listdir(imgs_path) if "jpg" in x]
print("number of images:", len(files))

original = load_img(files[5],  target_size=(image_width, image_height))

plt.imshow(original)
plt.show()

numpy_image = img_to_array(original)
images = np.expand_dims(numpy_image, axis=0)
print(images.shape)

processed_image = preprocess_input(images.copy())

img_features = feat_extractor.predict(processed_image)

print("features successfully extracted!")
print("number of image features:", img_features.size)
print(img_features)
