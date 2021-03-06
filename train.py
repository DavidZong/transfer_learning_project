import tensorflow as tf
import numpy as np
import inception
from inception import transfer_values_cache
import os
import time

# Open saved files
storage_path = 'storage_b_shuffled_seed1'
model = inception.Inception()
file_path_cache_train = os.path.join(storage_path, 'inception_image_train.pkl')
transfer_values_training = transfer_values_cache(cache_path=file_path_cache_train, model=model)
label_path = os.path.join(storage_path, 'labels.npz')

print('\nLoading Labels...')
proc_img_start_time = time.time()
labels_array = np.load(label_path)
labels = labels_array['arr_0']
proc_img_end_time = time.time()
print('done\n')
print('Processing took %s sec\n' % (proc_img_end_time - proc_img_start_time))

# Initialize variables for 3 layer network
#transfer_len = model.transfer_len
transfer_len = 2048
output_len = 89401


# Placeholder variables for the input and output
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, output_len], name='y_true')

# Placeholder for the phase, True if training, False if testing. For batchnorm
train = tf.placeholder(tf.bool)

# global variable to decide whether or not to regularize
# let's set this equal to False, because, as the Prof said, we were underfitting.
regularize = False

# helper function to make a weight variable
def weight_variable(name, shape):

    #initial = tf.contrib.layers.xavier_initializer()
    initial = tf.truncated_normal(shape, stddev=0.1)
    if regularize == True:
        # regularization term.  lambda
        #scale = 1e-4
        #scale = .25
        scale = .025

        return tf.get_variable(name, shape, initializer=initial, regularizer=tf.contrib.layers.l2_regularizer(scale))
    else:
        #return tf.get_variable(name, shape, initializer=initial)
        return tf.Variable(initial)

# helper function to make a bias variable
def bias_variable(shape):
    return tf.constant(0.1, shape=shape)


def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

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
# convolutional layer
#with tf.name_scope('conv1'):
#    W_conv1 = weight_variable('wconv1',[5, 5, 1, 32])
#    b_conv1 = bias_variable([32])

#    x_image = tf.reshape(x, [-1, 28, 28, 1])

#    input_1 = conv2d(x_image, W_conv1) + b_conv1



# Fully connected layer
with tf.name_scope('fc1'):
    with tf.name_scope('weight'):
        W_fc1 = weight_variable('wfc1',[transfer_len, 1024])
        #W_fc1 = weight_variable('wfc1', [np.size(input_1,1), 1024])
        variable_summaries(W_fc1)
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1)
    with tf.name_scope('net_input'):
        z_fc1 = tf.matmul(x, W_fc1) + b_fc1
        #z_fc1 = tf.matmul(input_1, W_fc1) + b_fc1
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
if regularize == True:
    regularization = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y_true) + regularization
else:
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_,labels = y_true)
loss = tf.reduce_mean(cross_entropy)

with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)
correct_prediction = tf.equal(tf.round(y_), y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#max_iter = 1500
max_iter = 9000
batchsize = 20


# flatten labels, convert to a non-one hot vector encoding (9000 1s or 0s).
f_labels = np.zeros([labels.shape[0], output_len])
for i in range(labels.shape[0]):
    f_labels[i] = labels[i].flatten()

# separate data set to training and test sets 70/30 split (roughly)
test_size = round(0.3 * labels.shape[0])
test_labels = f_labels[0:test_size]
train_labels = f_labels[test_size:]
test_data = transfer_values_training[0:test_size]
train_data = transfer_values_training[test_size:]



start_time = time.time()
print('\nStart time: ' + time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                        time.gmtime()) + '\n')
# Training loop
with tf.Session() as s:
    # setup summary writer
    tf.summary.scalar("loss", loss)
    summary_op = tf.summary.merge_all()
    test_summary = tf.summary.scalar("test_accuracy", accuracy)
    training_summary = tf.summary.scalar("training_accuracy", accuracy)
    saver = tf.train.Saver()
    result_dir = 'results/b_shuffled_lr-1e-4_seed1'
    summary_writer = tf.summary.FileWriter(result_dir, s.graph)
    s.run(tf.global_variables_initializer())

    # Loop optimization
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
            train_accuracy = s.run(training_summary,
                                     feed_dict={x: batch_x, y_true: batch_y, keep_prob: 1.0, train: 0})
            test_accuracy = s.run(test_summary,
                                      feed_dict={x: test_data, y_true: test_labels, keep_prob: 1.0, train: 0})
            summary_str = s.run(summary_op,
                                   feed_dict={x: batch_x, y_true: batch_y, keep_prob: 1.0, train: 0})
            summary_writer.add_summary(summary_str, i)
            summary_writer.add_summary(test_accuracy, i)
            summary_writer.add_summary(train_accuracy, i)
            summary_writer.flush()
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_true: batch_y, keep_prob: 1.0, train: 0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            test_accuracy = accuracy.eval(feed_dict={x: test_data, y_true: test_labels, keep_prob: 1.0, train: 0})
            print("step %d, test accuracy %g" % (i, test_accuracy))
    checkpoint_file = os.path.join(result_dir, 'final_checkpoint')
    saver.save(s, checkpoint_file)

end_time = time.time()
print('Total sec: %s sec' % (end_time-start_time))
print('Total min: %s min\n' % ((end_time-start_time)/60.0))
