import tensorflow as tf
import os
import inception
from inception import transfer_values_cache
import numpy as np
import matplotlib.pyplot as plt

# Load in trained model
checkpoint_dir = "results_lr10e4/"
s = tf.InteractiveSession()
meta_path = os.path.join(checkpoint_dir, "final_checkpoint.meta")
saver = tf.train.import_meta_graph(meta_path)
graph = tf.get_default_graph()
saver.restore(s, tf.train.latest_checkpoint(checkpoint_dir))

# Load in sample data
storage_path = 'storage_small'
model = inception.Inception()
file_path_cache_train = os.path.join(storage_path, 'inception_image_train.pkl')
transfer_values_training = transfer_values_cache(cache_path=file_path_cache_train, model=model)
label_path = os.path.join(storage_path, 'labels.npz')
print('Loading Labels...')
labels_array = np.load(label_path)
labels = labels_array['arr_0']
print('done')

# generate predictions
transfer_len = model.transfer_len
x = graph.get_tensor_by_name("x:0")
y_ = graph.get_tensor_by_name("softmax/net_input/Sigmoid:0")
train = graph.get_tensor_by_name("Placeholder:0")
keep_prob = graph.get_tensor_by_name("Placeholder_1:0")
prediction = s.run(y_, feed_dict={x: transfer_values_training, train:False, keep_prob:1})

# display image
example_index = 2
image = transfer_values_training[example_index]
label = labels[example_index]
predicted = np.reshape(prediction[example_index], (299,299))


fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(label)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(predicted)

# display with thresholding
threshold = 0.9
binary_mask = (predicted > threshold).astype(np.int_)

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(label)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(binary_mask)
