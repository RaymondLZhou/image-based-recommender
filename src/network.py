from tensorflow.keras.applications import vgg16

vgg_model = vgg16.VGG16(weights='imagenet')

vgg_model.summary()

