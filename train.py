import tensorflow as tf
import numpy as np
import inception
from inception import transfer_values_cache
import os
import time

# Open saved files
storage_path = 'storage_small'
model = inception.Inception()
file_path_cache_train = os.path.join(storage_path, 'inception_image_train.pkl')
transfer_values_training = transfer_values_cache(cache_path=file_path_cache_train, model=model)
label_path = os.path.join(storage_path, 'labels.npz')

print('\nLoading Labels...')
proc_img_start_time = time.time()
labels_array = np.load(label_path)
labels = labels_array['arr_0']
print('%s images loaded...' % len(labels))
proc_img_end_time = time.time()
print('done\n')
print('Processing took %s sec\n' % (proc_img_end_time - proc_img_start_time))

# Initialize variables for 3 layer network
transfer_len = model.transfer_len
output_len = 89401


# Placeholder variables for the input and output
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, output_len], name='y_true')

# Placeholder for the phase, True if training, False if testing. For batchnorm
train = tf.placeholder(tf.bool, name='is_training')

# helper function to make a weight variable
def weight_variable(name, shape):
    initial = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape, initializer=initial)

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
hidden_neurons = 512
print('Hidden neurons: %s' % hidden_neurons)
# Fully connected layer
with tf.name_scope('fc1'):
    with tf.name_scope('weight'):
        W_fc1 = weight_variable('wfc1', [transfer_len, hidden_neurons])
        variable_summaries(W_fc1)
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([hidden_neurons])
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
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
with tf.name_scope('softmax'):
    with tf.name_scope('weight'):
        W_fc2 = weight_variable('wfc2', [hidden_neurons, output_len])
        variable_summaries(W_fc2)
    with tf.name_scope('bias'):
        b_fc2 = bias_variable([output_len])
        variable_summaries(b_fc2)
    with tf.name_scope('net_input'):
        y_ = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='prediction')
        variable_summaries(y_)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y_true)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

learning_rate = 1e-4
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    print('\nUsing Adam Opt with learning rate %s\n' % learning_rate)
    #print('\nUsing Grad Descent with learning rate %s\n' % learning_rate)
correct_prediction = tf.equal(tf.round(y_), y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


max_iter = 1000
batchsize = 10
print('Batch size: %s\n' % batchsize)

# downstream, we do a 70:30 train:test split.  Unfortunately, our data is serially correlated.
# we must shuffle it to lose this correlation.
# we randomly shuffle the label matrices and transfer value matrices together

# reshape so we can combine labels and transfer values (we'll un-reshape later)
labels_tvs = np.c_[labels.reshape(len(labels),-1),
                               transfer_values_training.reshape(len(transfer_values_training),-1)]

# shuffle this combined matrix
np.random.shuffle(labels_tvs)

# "un-reshaping"
labels_shuffled = labels_tvs[:, :labels.size//len(labels)].reshape(labels.shape)
tv_shuffled = labels_tvs[:, labels.size//len(labels):].reshape(transfer_values_training.shape)

# flatten labels, convert to a non-one hot vector encoding (9000 1s or 0s).
f_labels = np.zeros([labels.shape[0], output_len])
for i in range(labels.shape[0]):
    # f_labels uses shuffled labels
    f_labels[i] = labels_shuffled[i].flatten()


# separate data set to training and test sets 70/30 split (roughly)
test_size = round(0.3 * labels.shape[0])
test_labels = f_labels[0:test_size]
train_labels = f_labels[test_size:]
test_data = tv_shuffled[0:test_size]
train_data = tv_shuffled[test_size:]


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
    result_dir = 'results_ad_hn512_lr1e4'
    print('\nResults written to %s\n' % result_dir)
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
            test_accuracy = s.run(test_summary,
                                     feed_dict={x: batch_x, y_true: batch_y, keep_prob: 1.0, train: 0})
            train_accuracy = s.run(training_summary,
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
