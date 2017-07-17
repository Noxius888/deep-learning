import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from CNN import create_cnn, load_data
from keras.datasets import mnist
from keras import backend as K

# Create the model and load the weights
model = create_cnn()
model.load_weights('cnn-mnist.h5')

# Visualize weights
W = model.layers[0].W.get_value(borrow=True)
print("W shape : ", W.shape)

plt.figure()
plt.title('Learned weights in the first convolutional layer')
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(W[i, 0, :, :], cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([]); plt.yticks([])
    plt.colorbar()

# Take one digit from the MNIST dataset
(X_train, Y_train), (X_test, Y_test), X_mean = load_data()
idx = 2989 # or any other digit
img = X_train[idx, :, :, :] 

plt.figure()
plt.title('Example of a digit')
plt.imshow(img[0, :, :] + X_mean[0, :, :], cmap=plt.cm.gray, interpolation='nearest')
plt.colorbar()

# Check the predicted output is correct
output = model.predict(img.reshape(1, 1, 28, 28), verbose=0)
prediction = output[0].argmax()
true_label = Y_train[idx, :].argmax()
print('Predicted digit:', prediction, '; True digit:', true_label)
plt.figure()
plt.title('Softmax probabilities')
plt.plot(output[0])

# Visualize convolutions
get_convolutions = K.function([model.layers[0].input, K.learning_phase()], [model.layers[1].output])
convolutions = get_convolutions([img.reshape(1, 1, 28, 28), 0])[0][0, :, :, :]

plt.figure()
plt.title('Activities in the convolutional layer')
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(convolutions[i, :, :], cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([]); plt.yticks([])
    plt.colorbar()

# Visualize Pooling
get_pooling = K.function([model.layers[0].input, K.learning_phase()], [model.layers[2].output])
poolings = get_pooling([img.reshape(1, 1, 28, 28), 0])[0][0, :, :, :]

plt.figure()
plt.title('Activities in the max-pooling layer')
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(poolings[i, :, :], cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks([]); plt.yticks([])
    plt.colorbar()

plt.show()
