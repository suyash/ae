# ae

Autoencoder implementations.

```
pipenv run python trainer/task.py  --job-dir $OUTPUT_DIR
```

creating a web model from generated exported models

```
pipenv run tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_node_names='output/BiasAdd' \
  --saved_model_tags=serve \
  $MODEL_DIR \
  $MODEL_DIR/web
```
