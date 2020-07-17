import tensorflow as tf
import time

import images
import model

def style_content_loss(outputs, extractor, layers, inputs, weights):
    style_outputs = outputs['style']
    content_outputs = outputs['content']

    content_image = inputs['content_image']
    style_image = inputs['style_image']

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    style_layers = layers['style_layers']
    content_layers = layers['content_layers']

    style_weight = weights['style_weight']
    content_weight = weights['content_weight']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_layers)

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_layers)
    
    loss = style_loss + content_loss

    return loss

@tf.function()
def train_step(image, optimizer, layers, inputs, weights, extractor):
    total_variation_weight = weights['total_variation_weight']

    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, extractor, layers, inputs, weights)
        loss += total_variation_weight * tf.image.total_variation(image)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    
    image.assign(images.clip_image(image))

def transfer_style(image, epochs, steps_per_epoch, optimizer, layers, inputs, weights):
    style_layers = layers['style_layers']
    content_layers = layers['content_layers']

    extractor = model.StyleContentModel(style_layers, content_layers)

    start = time.time()

    for n in range(epochs):
        start_epoch = time.time()

        print("Training epoch: {} out of {}".format(n+1, epochs))

        for i in range(steps_per_epoch):
            if (i+1) % 50 == 0:
                print("Step: {}".format(i+1))

            train_step(image, optimizer, layers, inputs, weights, extractor)
        
        end = time.time()

        print("Epoch time: {:.1f}s".format(end-start_epoch))
        print("Total time: {:.1f}s".format(end-start))
        print()
