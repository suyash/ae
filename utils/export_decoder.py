"""
export_decoder

exports the decoder half of an autoencoder SavedModel as a separate SavedModel.
"""

import argparse

import tensorflow as tf

def run(model_dir, layers):
    decoder = []

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_dir)

        for i in range(len(layers) - 1):
            k = sess.run(sess.graph.get_tensor_by_name("decoder_%d/kernel:0" % (i + 1)))
            b = sess.run(sess.graph.get_tensor_by_name("decoder_%d/bias:0" % (i + 1)))
            decoder.append((k, b))

        k = sess.run(sess.graph.get_tensor_by_name("logits/kernel:0"))
        b = sess.run(sess.graph.get_tensor_by_name("logits/bias:0"))

        decoder.append((k, b))

    inputs = tf.placeholder(tf.float32, [None, layers[-1]])
    layer = inputs

    for i in range(len(layers) - 1):
        with tf.variable_scope("decoder_%d" % (i + 1)):
            kernel = tf.Variable(decoder[i][0], name="kernel")
            bias = tf.Variable(decoder[i][1], name="bias")
            layer = tf.nn.bias_add(tf.matmul(layer, kernel), bias)
            layer = tf.nn.elu(layer)

    with tf.variable_scope("logits"):
        kernel = tf.Variable(decoder[-1][0], name="kernel")
        bias = tf.Variable(decoder[-1][1], name="bias")
        logits = tf.nn.bias_add(tf.matmul(layer, kernel), bias)
        logits = tf.nn.elu(logits)

    outputs = tf.sigmoid(logits, name="outputs")

    signature_def = tf.saved_model.signature_def_utils.predict_signature_def(
        { "embeddings": inputs },
        { "predictions": outputs },
    )

    builder = tf.saved_model.builder.SavedModelBuilder("%s/decoder" % model_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        builder.add_meta_graph_and_variables(
            sess,
            [ tf.saved_model.tag_constants.SERVING ],
            signature_def_map={ "serving_default": signature_def },
            strip_default_attrs=True,
        )

    builder.save()

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
    )

    HYPERPARAMS, _ = parser.parse_known_args()
    run(**HYPERPARAMS.__dict__)

if __name__ == "__main__":
    main()
