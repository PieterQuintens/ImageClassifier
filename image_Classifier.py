from keras.datasets import cifar10
import keras.utils as utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.constraints import max_norm
from keras.optimizers import SGD

# Load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Model
# I made a very simple model with just 1 convolution layer, 1 maxpool layer
# and 2 dense layers with a dropout layer in between.

model = Sequential()
# padding is set to same so the output image is the same as the input
# kernel_constraint normalises values if they go above 3

# convolution layer
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu', padding='same',
                 kernel_constraint=max_norm(3)))
# maxpooling layer (comes after convolution layers)
# Because the pool size is (2, 2) the output image is going to be half the size of the input
model.add(MaxPooling2D(pool_size=(2, 2)))

# flatten everything out to get 1 row of features instead of a matrix
model.add(Flatten())

# dense layer: units define the output shape (Array of length 512)
model.add(Dense(units=512, activation='relu', kernel_constraint=max_norm(3)))
# dropout layer to prevent overfitting (only in training not in testing) 0.5 to drop half the neurons
model.add(Dropout(rate=0.5))

# because we only want the probability of 10 output categories
# Because we are working with multiple probabilities we use the Softmax activation function
# If we want the probability between 2 things we can use the Sigmoid activation function
model.add(Dense(units=10, activation='softmax'))

# the model is now done, the only thing left is the compile and train step

# if the learning rate of the SGD is lower the model will get more accurate but you will need a lot more epochs and time
# to get a decent result
model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Train step
model.fit(x=x_train, y=y_train, epochs=30, batch_size=32)

# Saves the trained model
model.save(filepath="Image_Classifier.h5")
