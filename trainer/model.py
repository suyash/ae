import tensorflow as tf

def model_fn(features, labels, mode, params):
    kernel_initializer = tf.variance_scaling_initializer()
    kernel_regularizer = tf.contrib.layers.l2_regularizer(params["weight_decay"])

    if isinstance(features, dict):
        features = features["image"]

    features = tf.reshape(features, [-1, 28 * 28])
    features = tf.cast(features, tf.float32) / 255.0

    layer = features

    for (i, l) in enumerate(params["layers"][:-1]):
        layer = tf.layers.dense(
            layer,
            l,
            activation=tf.nn.elu,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name="encoder_%d" % (i + 1),
        )

    layer = tf.layers.dense(
        layer,
        params["layers"][-1],
        activation=tf.nn.elu,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name="embedding",
    )

    for (i, l) in enumerate(reversed(params["layers"][:-1])):
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

    if mode == tf.estimator.ModeKeys.PREDICT:
        probs = tf.nn.sigmoid(logits)
        predictions = { "prediction": probs }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={ "predict": tf.estimator.export.PredictOutput(predictions) },
        )

    loss = tf.losses.sigmoid_cross_entropy(features, logits)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

    loss = tf.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(params["learning_rate"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
