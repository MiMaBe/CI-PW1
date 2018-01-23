import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist.mnist import NUM_CLASSES
from cnn_model import cnn_model


tf.logging.set_verbosity(tf.logging.INFO)

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

data = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = data.train.images[: 5500]
train_labels = np.asarray(data.train.labels, dtype=np.int32)[: 5500]
eval_data = data.test.images[: 1000]
eval_labels = np.asarray(data.test.labels, dtype=np.int32)[: 1000]

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model, model_dir="/tmp/mnist_convnet_model")

tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=12000,
    hooks=[logging_hook])

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

sess.close()
