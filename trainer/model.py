import tensorflow as tf

def encoder(features, layers, kernel_initializer, kernel_regularizer):
    layer = features

    for (i, l) in enumerate(layers[:-1]):
        layer = tf.layers.dense(
            layer,
            l,
            activation=tf.nn.elu,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="encoder_%d" % (i + 1),
        )

    embedding = tf.layers.dense(
        layer,
        layers[-1],
        activation=tf.nn.elu,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name="embedding",
    )

    return embedding

def decoder(features, layers, kernel_initializer, kernel_regularizer):
    layer = features

    for (i, l) in enumerate(reversed(layers[:-1])):
        layer = tf.layers.dense(
            layer,
            l,
            activation=tf.nn.elu,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="decoder_%d" % (i + 1),
        )

    logits = tf.layers.dense(
        layer,
        28 * 28,
        activation=None,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name="logits",
    )

    return logits

def add_noise(image, noise_factor):
    noise = noise_factor * tf.random_uniform(image.shape)
    return tf.clip_by_value(tf.add(image, noise), 0.0, 1.0)

def model_fn(features, labels, mode, params):
    kernel_initializer = tf.variance_scaling_initializer()
    kernel_regularizer = tf.contrib.layers.l2_regularizer(params["weight_decay"])

    if isinstance(features, dict):
        features = features["image"]

    features = tf.reshape(features, [-1, 28 * 28])
    features = tf.cast(features, tf.float32) / 255.0

    if params["noise_factor"] > 0:
        images = tf.map_fn(lambda x: add_noise(x, params["noise_factor"]), features, back_prop=False)
    else:
        images = features

    embeddings = encoder(images, params["layers"], kernel_initializer, kernel_regularizer)
    logits = decoder(embeddings, params["layers"], kernel_initializer, kernel_regularizer)

    if mode == tf.estimator.ModeKeys.PREDICT:
        outputs = tf.nn.sigmoid(logits, name="outputs")
        predictions = { "predictions": outputs, "embeddings": embeddings }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={ "serving_default": tf.estimator.export.PredictOutput(predictions) },
        )

    loss = tf.losses.sigmoid_cross_entropy(features, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    loss = tf.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
