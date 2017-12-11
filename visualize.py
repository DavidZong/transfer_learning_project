import tensorflow as tf
import os
import inception
from inception import transfer_values_cache
import numpy as np
import matplotlib.pyplot as plt

# Load in trained model
checkpoint_dir = "results/mam_largeset_9000iter_20batch_pre-shuffled"
s = tf.InteractiveSession()
meta_path = os.path.join(checkpoint_dir, "final_checkpoint.meta")
saver = tf.train.import_meta_graph(meta_path)
tf.reset_default_graph() #need to do this otherwise getting graph will crash the program
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

# execution:  if you get an error "the name x:0 refers to a Tensor...", then run the lines
# 'checkpoint_dir'  THROUGH 'saver'
# then skip the reset graph line then then execute the next two lines:
# graph AND saver.restore


x = graph.get_tensor_by_name("x:0")
#y_ = graph.get_tensor_by_name("softmax/net_input/prediction:0") # change prediction to Sigmoid if you get error
y_ = graph.get_tensor_by_name("softmax/net_input/Sigmoid:0") # change prediction to Sigmoid if you get error
#train = graph.get_tensor_by_name("is_training:0") # change to Placeholder if you get error
train = graph.get_tensor_by_name("Placeholder:0") # change to Placeholder if you get error
#keep_prob = graph.get_tensor_by_name("keep_prob:0") # change to Placeholder_1 if you get error
keep_prob = graph.get_tensor_by_name("Placeholder_1:0") # change to Placeholder_1 if you get error
prediction = s.run(y_, feed_dict={x: transfer_values_training, train:False, keep_prob:1})

# display image
def plot_images_at_index(example_index,plot_phase = False):
    label = labels[example_index]
    predicted = np.reshape(prediction[example_index], (299, 299))

    # display with thresholding
    threshold = 0.5
    binary_mask = (predicted > threshold).astype(np.int_)

    if plot_phase == False:

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

    else:
        plottable_image = images[example_index] #np.reshape(prediction[example_index], (299, 299))
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(plottable_image)
        plt.axis('off')
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(label)
        plt.axis('off')
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(predicted)
        plt.axis('off')
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(binary_mask)
        plt.axis('off')
