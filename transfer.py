import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import inception
from inception import transfer_values_cache
import os

img = mpimg.imread("phase.png")

crop = img[0:299, 0:299, :]

images = np.empty((3, 299, 299, 3))
images[0,:,:,:] = crop
images[1,:,:,:] = np.fliplr(crop)
images[2,:,:,:] = np.flipud(crop)

images = images * 255
label = mpimg.imread("feature_1.png")

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