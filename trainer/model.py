import tensorflow as tf

def linear_encoder(features, layers, kernel_initializer, kernel_regularizer):
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

def linear_decoder(features, layers, kernel_initializer, kernel_regularizer):
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

def conv_encoder(features, layers, kernel_initializer, kernel_regularizer):
    layer = features

    for (i, (filters, kernel_size, pool_size)) in enumerate(layers[:-1]):
        layer = tf.layers.conv2d(layer, filters, kernel_size, activation="relu", padding="same", name="encoder_%d" % (i + 1))
        layer = tf.layers.max_pooling2d(layer, pool_size, strides=pool_size, padding="same", name="encoder_%d_pool" % (i + 1))

    embedding = tf.layers.conv2d(layer, layers[-1][0], layers[-1][1], activation="relu", padding="same", name="embedding")
    embedding = tf.layers.max_pooling2d(embedding, layers[-1][2], strides=layers[-1][2], padding="same", name="embedding_pool")
    return embedding

def conv_decoder(features, layers, kernel_initializer, kernel_regularizer):
    layer = features

    for (i, (filters, kernel_size, pool_size)) in enumerate(reversed(layers)):
        layer = tf.layers.conv2d_transpose(layer, filters, kernel_size, strides=pool_size, activation="relu", padding="same", name="decoder_%d" % (i + 1))

    logits = tf.layers.conv2d_transpose(layer, 1, (3, 3), activation=None, padding="same", name="logits")
    logits = logits[:, 2:30, 2:30, :]
    return logits

def add_noise(image, noise_factor):
    noise = noise_factor * tf.random_uniform(image.shape)
    return tf.clip_by_value(tf.add(image, noise), 0.0, 1.0)

def model_fn(features, labels, mode, params):
    kernel_initializer = tf.variance_scaling_initializer()
    kernel_regularizer = tf.contrib.layers.l2_regularizer(params["weight_decay"])

    if isinstance(features, dict):
        features = features["image"]

    tf.summary.image("input", tf.reshape(features, [-1, 28, 28, 1]), max_outputs=10)

    if params["linear"]:
        features = tf.reshape(features, [-1, 28 * 28])
    else:
        features = tf.reshape(features, [-1, 28, 28, 1])

    features = tf.cast(features, tf.float32) / 255.0

    if params["noise_factor"] > 0:
        images = tf.map_fn(lambda x: add_noise(x, params["noise_factor"]), features, back_prop=False)
    else:
        images = features

    if params["linear"]:
        embeddings = linear_encoder(images, params["layers"], kernel_initializer, kernel_regularizer)
        logits = linear_decoder(embeddings, params["layers"], kernel_initializer, kernel_regularizer)
    else:
        embeddings = conv_encoder(images, params["layers"], kernel_initializer, kernel_regularizer)
        logits = conv_decoder(embeddings, params["layers"], kernel_initializer, kernel_regularizer)

    outputs = tf.nn.sigmoid(logits, name="outputs")
    tf.summary.image("output", tf.reshape(outputs, [-1, 28, 28, 1]), max_outputs=10)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = { "predictions": outputs, "embeddings": embeddings }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={ "serving_default": tf.estimator.export.PredictOutput(predictions) },
        )

    loss = tf.losses.sigmoid_cross_entropy(features, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir="%s/eval" % params["job_dir"],
            summary_op=tf.summary.merge_all(),
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, evaluation_hooks=[ summary_hook ])

    loss = tf.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
