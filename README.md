# ae

Autoencoder implementations.

To train a Convolutional Model.

```
pipenv run python trainer/task.py  --job-dir $OUTPUT_DIR
```

To train a Linear Model.

```
pipenv run python trainer/task.py --job-dir $OUTPUT_DIR --linear
```

To train a variational model, pass `--variational` to either commands.

---

Creating a web model from generated exported models

```
pipenv run tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='outputs' \
    --saved_model_tags=serve \
    $MODEL_DIR \
    $MODEL_DIR/web
```

## Other Art

- https://www.jeremyjordan.me/autoencoders/

- https://www.jeremyjordan.me/variational-autoencoders/

- https://blog.keras.io/building-autoencoders-in-keras.html
