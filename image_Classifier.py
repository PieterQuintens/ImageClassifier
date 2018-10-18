from keras.datasets import cifar10
import keras.utils as utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.constraints import max_norm
from keras.optimizers import SGD

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = Sequential()
# padding to make sure the convoluted image is the same size as the input image, kernel_constraint makes
# sure if the values go above 3, the value is normalised to 3 so the value doesn't get to large
# convolution layer
model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu', padding='same',
                 kernel_constraint=max_norm(3)))
# maxpooling layer (because previous layer was convolution)
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten everything out to get 1 row of features instead of a matrix
model.add(Flatten())
# dense layer: units = accuracy
model.add(Dense(units=512, activation='relu', kernel_constraint=max_norm(3)))
# dropout layer to prevent overfitting (only in training not in testing) 0.5 to drop half the neurons
model.add(Dropout(rate=0.5))
# because we only want the probability of 10 output categories and because we are working with
# probabilities we use the softmax activation
model.add(Dense(units=10, activation='softmax'))

# the model is now done, the only thing left is the compile and train step

# if the learning rate of the SGD is lower the model will get more accurate but you will need a lot more epochs and time
# to get a decent result
model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=30, batch_size=32)

model.save(filepath="Image_Classifier.h5")

