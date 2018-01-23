'''
Created on Jan 15, 2018

@author: mimabe
'''
import tensorflow as tf

def cnn_model(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
    l1_conv = tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    l1_pooling = tf.layers.max_pooling2d(inputs=l1_conv, pool_size=[2, 2], strides=2)
    
    l2_conv = tf.layers.conv2d(inputs=l1_pooling,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)
    l2_pooling = tf.layers.max_pooling2d(inputs=l2_conv, pool_size=[2, 2], strides=2)
    l2_pooling_flat = tf.reshape(l2_pooling, [-1, 7 * 7 * 64])
    
    dense = tf.layers.dense(inputs=l2_pooling_flat, units=1024, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    logits = tf.layers.dense(inputs=dropout, units=10)
    
    predictions = {"classes": tf.argmax(input=logits, axis=1),"probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]),
        "precision": tf.metrics.precision(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)