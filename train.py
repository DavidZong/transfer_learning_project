import tensorflow as tf
import numpy as np
import inception
from inception import transfer_values_cache
import os

# Open saved files
model = inception.Inception()
file_path_cache_train = os.path.join("storage", 'inception_image_train.pkl')
transfer_values_training = transfer_values_cache(cache_path=file_path_cache_train, images=images, model=model)

# Initialize variables for 3 layer network
transfer_len = model.transfer_len
output_len = 89401
# Placeholder variables for the input and output
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, output_len], name='y_true')
# Placeholder for the phase, True if training, False if testing. For batchnorm
train = tf.placeholder(tf.bool)

# helper function to make a weight variable
def weight_variable(name, shape):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer())

# helper function to make a bias variable
def bias_variable(shape):
    return tf.constant(0.1, shape=shape)

# helper function to attach variable summary stats
def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# Define the model
# Fully connected layer
with tf.name_scope('fc1'):
    with tf.name_scope('weight'):
        W_fc1 = weight_variable('wfc1', [transfer_len, 1024])
        variable_summaries(W_fc1)
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1)
    with tf.name_scope('net_input'):
        z_fc1 = tf.matmul(x, W_fc1) + b_fc1
        variable_summaries(z_fc1)
    with tf.name_scope('batch_norm'):
        bn_fc1 = tf.layers.batch_normalization(z_fc1, training=train)
    with tf.name_scope('activation'):
        h_fc1 = tf.nn.relu(bn_fc1)
        variable_summaries(h_fc1)

# Apply dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
with tf.name_scope('softmax'):
    with tf.name_scope('weight'):
        W_fc2 = weight_variable('wfc2', [1024, output_len])
        variable_summaries(W_fc2)
    with tf.name_scope('bias'):
        b_fc2 = bias_variable([output_len])
        variable_summaries(b_fc2)
    with tf.name_scope('net_input'):
        y_ = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        variable_summaries(y_)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y_true)
loss = tf.reduce_mean(cross_entropy)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.round(y_), y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

max_iter = 10000
batchsize = 10

# flatten labels
f_labels = np.zeros([labels.shape[0], output_len])
for i in range(labels.shape[0]):
    f_labels[i] = labels[i].flatten()

# separate data set to training and test sets 70/30 split (roughly)
test_size = round(0.3 * n_examples)
test_labels = f_labels[0:test_size]
train_labels = f_labels[test_size:]
test_data = transfer_values_training[0:test_size]
train_data = transfer_values_training[test_size:]

# Training loop
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    batch_x = np.zeros([batchsize, model.transfer_len])
    batch_y = np.zeros([batchsize, output_len])
    for i in range(max_iter):
        nsamples = train_labels.shape[0]
        perm = np.arange(nsamples)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_x[j, :] = train_data[perm[j], :]
            batch_y[j, :] = train_labels[perm[j], :]
        s.run(train_step, feed_dict={x: batch_x, y_true: batch_y, keep_prob: 0.5, train: 1})

        # Test the training accuracy every so often
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_true: batch_y, keep_prob: 1.0, train: 0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            test_accuracy = accuracy.eval(feed_dict={x: test_data, y_true: test_labels, keep_prob: 1.0, train: 0})
            print("step %d, test accuracy %g" % (i, test_accuracy))