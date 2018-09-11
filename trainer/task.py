import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

from model import model_fn

def run(
    max_steps,
    batch_size,
    layers,
    learning_rate,
    weight_decay,
    noise_factor,
    job_dir,
):
    config = tf.estimator.RunConfig(
        model_dir=job_dir,
        save_checkpoints_steps=500,
    )

    estimator = tf.estimator.Estimator(
        model_fn,
        config=config,
        params={
            "layers": layers,
            "weight_decay": weight_decay,
            "learning_rate": learning_rate,
            "noise_factor": noise_factor,
            "job_dir": job_dir,
        },
    )

    (train_data, _), (eval_data, _) = mnist.load_data()

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        train_data,
        num_epochs=None,
        shuffle=True,
        batch_size=150,
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        eval_data,
        num_epochs=1,
        shuffle=False,
    )

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=max_steps)

    serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "image": tf.placeholder(tf.uint8, [None, 28, 28]),
    })
    exporter = tf.estimator.LatestExporter("model", serving_input_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, throttle_secs=30, exporters=[ exporter ])

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

# https://stackoverflow.com/a/12117065/3673043
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%s not in range [0.0, 1.0)" % x)
    return x

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-size',
        help="""Batch size for training and eval steps""",
        default=150,
        type=int
    )

    parser.add_argument(
        '--max-steps',
        help="""Maximum number of steps""",
        default=5000,
        type=int
    )

    parser.add_argument(
        '--layers',
        help="""Sizes of encoding layers""",
        nargs="+",
        default=[512, 128, 32],
        type=int
    )

    parser.add_argument(
        '--learning-rate',
        help="""Learning rate value for the optimizers""",
        default=0.001,
        type=float
    )

    parser.add_argument(
        '--weight-decay',
        help="""L2 regularizer weight decay""",
        default=1e-5,
        type=float
    )

    parser.add_argument(
        '--noise-factor',
        help="""noise factor to corrupt the input with""",
        default=0.0,
        type=restricted_float,
    )

    parser.add_argument(
        '--job-dir',
        help="""Local/GCS location to write checkpoints and export models""",
        required=True
    )

    tf.logging.set_verbosity(tf.logging.INFO)

    HYPERPARAMS, _ = parser.parse_known_args()
    run(**HYPERPARAMS.__dict__)

if __name__ == "__main__":
    main()
