import tensorflow as tf

content_name = 'tubingen'
style_name = 'Picasso'

epochs = 3
steps_per_epoch = 1

optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30
weights = dict(style_weight=style_weight, content_weight=content_weight, total_variation_weight=total_variation_weight)
