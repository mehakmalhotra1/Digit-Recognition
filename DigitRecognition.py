# Download the MNIST dataset directly from tensorflow and keras API
import tensorflow as tf

# x_train and x_test parts contain greyscale RGB codes (from 0 to 255)
# while y_train and y_test parts contains labels from 0 to 9 which
# represents which number they actually are. 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# To visualize these numbers, we can get help from matplotlib.
import matplotlib.pyplot as plt
%matplotlib inline 
img_index = 7776
print(y_train[img_index])
plt.imshow(x_train[img_index], cmap='Greys')

# the shape of the dataset to channel it to the convolutional neural network
x_train.shape

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(4,4), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays to 1D arrays for fully connected layers 
model.add(Dense(150, activation=tf.nn.relu))
model.add(Dropout(0.3)) # Dropout layers fight with the overfitting by disregarding some of the neurons while training 
model.add(Dense(10,activation=tf.nn.softmax))

# Setting an optimizer with a given loss function which uses a metric 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Fitting the model by using our train data
model.fit(x=x_train,y=y_train, epochs=10)

# Evaluating the trained model with x_test and y_test
model.evaluate(x_test, y_test)

#Result
img_idx = 9957
plt.imshow(x_test[img_idx].reshape(28, 28),cmap='Greys')
pred = model.predict(x_test[img_idx].reshape(1, 28, 28, 1))
print(pred.argmax())