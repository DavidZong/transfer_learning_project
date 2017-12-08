import tensorflow as tf
import os
import inception
from inception import transfer_values_cache
import numpy as np
import matplotlib.pyplot as plt

# Load in trained model
checkpoint_dir = "results/m_long"
s = tf.InteractiveSession()
meta_path = os.path.join(checkpoint_dir, "final_checkpoint.meta")
saver = tf.train.import_meta_graph(meta_path)
graph = tf.get_default_graph()
saver.restore(s, tf.train.latest_checkpoint(checkpoint_dir))

# Load in sample data
storage_path = 'storage_small_m'
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
y_ = graph.get_tensor_by_name("softmax/net_input/prediction:0") # change prediction to Sigmoid if you get error
train = graph.get_tensor_by_name("is_training:0") # change to Placeholder if you get error
keep_prob = graph.get_tensor_by_name("keep_prob:0") # change to Placeholder_1 if you get error
prediction = s.run(y_, feed_dict={x: transfer_values_training, train:False, keep_prob:1})

# display image
def plot_images_at_index(example_index):
    label = labels[example_index]
    predicted = np.reshape(prediction[example_index], (299, 299))

    # display with thresholding
    threshold = 0.5
    binary_mask = (predicted > threshold).astype(np.int_)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(label)
    plt.axis('off')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(predicted)
    plt.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(binary_mask)
    plt.axis('off')

