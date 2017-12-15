import tensorflow as tf
import os
import inception
from inception import transfer_values_cache
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

# Load in trained model
checkpoint_dir = 'results/b_shuffled_lr-1e-4_seed1'
s = tf.InteractiveSession()
meta_path = os.path.join(checkpoint_dir, "final_checkpoint.meta")
saver = tf.train.import_meta_graph(meta_path)
tf.reset_default_graph() #need to do this otherwise getting graph will crash the program
graph = tf.get_default_graph()
saver.restore(s, tf.train.latest_checkpoint(checkpoint_dir))

# Load in sample data
storage_path = 'storage_b_shuffled_seed1'
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
    confusion_matrix = np.abs((predicted - label))
    # jaccard index
    ji = sklearn.metrics.jaccard_similarity_score(np.int64(np.ndarray.flatten(label)),np.int64(np.ndarray.flatten(binary_mask)))
    print('Jaccard index is %s' % (np.around(ji,3)))
    if plot_phase == False:

        fig = plt.figure()
        plt.axis('off')
        plt.title('Jaccard index = %s' % (np.around(ji,3)))
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(label)
        plt.axis('off')
        plt.title('Ground Truth')
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(predicted)
        plt.axis('off')
        plt.title('Prediction')
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(binary_mask)
        plt.axis('off')
        plt.title('Thresholded Prediction')
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(confusion_matrix)
        plt.axis('off')
        plt.title('Confusion Matrix')

    else:
        plottable_image = images[example_index] #np.reshape(prediction[example_index], (299, 299))
        fig = plt.figure()
        plt.axis('off')
        plt.title('Jaccard index = %s' % (np.around(ji,3)))
        ax1 = fig.add_subplot(1, 5, 1)
        ax1.imshow(plottable_image)
        plt.axis('off')
        plt.title('Microscopy Image')
        ax2 = fig.add_subplot(1, 5, 2)
        ax2.imshow(label)
        plt.axis('off')
        plt.title('Ground Truth')
        ax3 = fig.add_subplot(1, 5, 3)
        ax3.imshow(predicted)
        plt.axis('off')
        plt.title('Prediction')
        ax4 = fig.add_subplot(1, 5, 4)
        ax4.imshow(binary_mask)
        plt.axis('off')
        plt.title('Thresholded Prediction')
        ax5 = fig.add_subplot(1, 5, 5)
        ax5.imshow(confusion_matrix)
        plt.axis('off')
        plt.title('Confusion Matrix')
