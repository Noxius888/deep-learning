from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# Load the data set
data = np.loadtxt('linear.data')

# Separate the input from the output
X = data[:, 0:-1]
Y = data[:, -1]
N, d = X.shape

# Separate the positive from the negative class
positive_class = X[Y==1., :]
negative_class = X[Y==-1., :]

# Visualize the dataset
def visualize(w, b):
    # Plot the positive and negative examples
    plt.plot(positive_class[:, 0], positive_class[:, 1], '.r')
    plt.plot(negative_class[:, 0], negative_class[:, 1], '.b')
    # Plot the hyperplane
    if w[1] != 0.0: # the hyperplane is otherwise vertical, would crash
        X = np.linspace(0., 1., 100)
        Y = -(w[0]*X + b)/w[1]
        plt.plot(X, Y)
    plt.show()


# Compute the error
def compute_error(w, b):
    nb_error = 0
    gamma = 1000.
    for i in range(N):
        scalar_product = np.dot(X[i, :], w) + b
        gamma = min(gamma, Y[i]*scalar_product)
        if np.sign(scalar_product) != Y[i]:
            nb_error += 1
            
    return nb_error/N, gamma
    
# Parameters
eta = 0.1

# Weight vector and bias
w = np.zeros(d)
b = 0.

# Perceptron algorithm
gamma = -1.
epoch = 1
while True:
    # Make a copy of the weight vector
    old_w = w.copy()
    # Iterate over all training examples
    for i in range(N):
        # Prediction of the hypothesis
        h_i = np.sign(np.dot(X[i, :], w) + b)
        # Update the weight
        w += eta * (Y[i] - h_i) * X[i, :]
        # Update the bias
        b += eta * (Y[i] - h_i) 

    # Calculate the functional margin
    error, gamma = compute_error(w, b)
    print('Epoch:', epoch)
    print('w:', w)
    print('b', b)
    print('Error:', error)
    print('Functional margin:', gamma)
    print ('Weight variation:', np.linalg.norm(w-old_w)/np.linalg.norm(old_w))
    epoch += 1
    if np.linalg.norm(w-old_w) < 0.01* np.linalg.norm(old_w):
        break

print('Norm of the weight vector:', np.linalg.norm(w))
visualize(w, b)

