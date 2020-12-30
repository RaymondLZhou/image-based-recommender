import time

import tensorflow as tf

import images
import model


def transfer_style(image, epochs, steps_per_epoch, optimizer, layers, inputs, weights):
    @tf.function()
    def train_step():
        def compute_style_content_loss():
            style_outputs = outputs['style']
            content_outputs = outputs['content']

            content_image = inputs['content_image']
            style_image = inputs['style_image']

            style_targets = extractor(style_image)['style']
            content_targets = extractor(content_image)['content']

            style_weight = weights['style_weight']
            content_weight = weights['content_weight']

            style_loss = tf.add_n(
                [tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
            style_loss *= style_weight / len(style_layers)

            content_loss = tf.add_n(
                [tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in
                 content_outputs.keys()])
            content_loss *= content_weight / len(content_layers)

            return style_loss + content_loss

        total_variation_weight = weights['total_variation_weight']

        with tf.GradientTape() as tape:
            outputs = extractor(image)
            loss = compute_style_content_loss()
            loss += total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])

        image.assign(images.clip_image(image))

    style_layers = layers['style_layers']
    content_layers = layers['content_layers']

    extractor = model.StyleContentModel(style_layers, content_layers)

    start = time.time()

    for epoch in range(epochs):
        start_epoch = time.time()

        print("Training epoch: {} out of {}".format(epoch + 1, epochs))

        for step in range(steps_per_epoch):
            if (step + 1) % 50 == 0:
                print("Step: {}".format(step + 1))

            train_step()

        end = time.time()

        print("Epoch time: {:.1f}s".format(end - start_epoch))
        print("Total time: {:.1f}s".format(end - start))
        print()
