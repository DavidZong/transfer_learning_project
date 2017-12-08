import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import inception
from inception import transfer_values_cache
import os

# Specify folder
storage_path = 'storage_small_m'
if not os.path.exists(storage_path):
    os.makedirs(storage_path)

# Load in image and label
img = mpimg.imread("phase_mcf10a.png")
label = mpimg.imread("feature_1_mcf10a.png")
# Stack the image and label
cat = np.dstack((img,label))

# one loop is a crop, produces 8 examples
loops = 4
n_examples = 8 * loops

# generate a set of unique coordinates to crop from
max_x = cat.shape[0]-299
max_y = cat.shape[1]-299
cord_set = set()
while len(cord_set) < loops:
    x, y = np.random.choice(max_x - 1), np.random.choice(max_y - 1)
    cord_set.add((x, y))
cord_list = list(cord_set)
# Make crops, flip, rotate and store in a 4D vector
i = 0
j = 0
images = np.empty((n_examples, 299, 299, 4))
while i < n_examples:
    if i % 100 == 0:
        print("crop images loop %d" % (i))
    x, y = cord_list[j]
    j = j + 1
    crop = cat[x:x+299, y:y+299, :]
    images[0+i, :, :, :] = crop
    images[1+i, :, :, :] = np.rot90(crop, 1)
    images[2+i, :, :, :] = np.rot90(crop, 2)
    images[3+i, :, :, :] = np.rot90(crop, 3)

    lr = np.fliplr(crop)
    images[4+i, :, :, :] = lr
    images[5+i, :, :, :] = np.rot90(lr, 1)
    images[6+i, :, :, :] = np.rot90(lr, 2)
    images[7+i, :, :, :] = np.rot90(lr, 3)

    i = i + 8

labels = images[:,:,:,3]
# save the labels so they can be used later, the ordering is the same as the images
print("saving labels...")
label_path = os.path.join(storage_path, 'labels')
np.savez_compressed(label_path, labels)
print("done.")
images = images[:,:,:,0:3]
images = images * 255

# calculate the transfer values using Inception, or load if already done
model = inception.Inception()
file_path_cache_train = os.path.join(storage_path, 'inception_image_train.pkl')
transfer_values_training = transfer_values_cache(cache_path=file_path_cache_train, images=images, model=model)
