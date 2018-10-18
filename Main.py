from keras.datasets import cifar10
import matplotlib.pyplot as plt
import keras.utils as utils
import numpy as np


# To flatten matrices to straight arrays (easier to use in java apps)

def reshape_image(input_image_arrays):
    output_array = []
    for image_array in input_image_arrays:
        output_array.append(image_array.reshape(-1))
    return np.asarray(output_array)


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

first_image = train_images[8]

# print(first_image[0])
# plt.imshow(first_image)
# plt.show()
# train_labels = utils.to_categorical(train_labels)
# max_index = np.argmax(train_labels[0])
# print(labels_array[max_index])

labels_array = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)

train_images = train_images.astype('float32')
train_images = train_images / 255
test_images = test_images.astype('float32')
test_images = test_images / 255





