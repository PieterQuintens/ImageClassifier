from keras.datasets import cifar10
import keras.utils as utils
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.image as im
import matplotlib.pyplot as plt

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32') / 255.0
y_test = utils.to_categorical(y_test)

model = load_model(filepath="Image_Classifier_30_epochs.h5")

results = model.evaluate(x=x_test, y=y_test)

print("Test loss:", results[0])
print("Test accuracy:", results[1])

image = x_test[107]

# img = Image.open("Testimage.jpg")
# img_array = np.asarray(img)
# img_array = img_array.astype('float32') /255.0
# img.show()

# print(img_array)
# print(x_test[0])


test_image_data = np.asarray([image])

prediction = model.predict(x=test_image_data)
# prediction = model.predict(x=img_array)

max_index = np.argmax(prediction[0])
print(prediction)
print(np.argmax(prediction))
print("Prediction: ", labels[max_index])



# plt.imshow(image)
# plt.show()lbgb