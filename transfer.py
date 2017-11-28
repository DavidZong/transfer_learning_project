import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import inception
from inception import transfer_values_cache
import os

# Load in image and label
img = mpimg.imread("phase.png")
label = mpimg.imread("feature_1.png")
# Stack the image and label
cat = np.dstack((img,label))

# one loop is a crop, produces 12 examples
loops = 4
n_examples = 12 * loops

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

    ud = np.flipud(crop)
    images[8+i, :, :, :] = ud
    images[9+i, :, :, :] = np.rot90(ud, 1)
    images[10+i, :, :, :] = np.rot90(ud, 2)
    images[11+i, :, :, :] = np.rot90(ud, 3)

    i = i + 12
labels = images[:,:,:,3]
# save the labels so they can be used later, the ordering is the same as the images
np.savez_compressed('storage/labels', labels)
images = images[:,:,:,0:3]
images = images * 255


model = inception.Inception()

file_path_cache_train = os.path.join("storage", 'inception_image_train.pkl')

transfer_values_training = transfer_values_cache(cache_path=file_path_cache_train, images=images, model=model)
print(transfer_values_training.shape)


def plot_transfer_values(i):
    print("Input image:")

    # Plot the i'th image from the test-set.
    plt.imshow(images[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values_training[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

plot_transfer_values(i=2)