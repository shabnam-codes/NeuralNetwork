import random
import numpy as np
import time 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical, plot_model

def to_categorical_np(y, num_classes=10):
    y = y.astype(int)
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

x,y = fetch_openml('mnist_784', version = 1, return_X_y=True)
x = (x/255).astype('float32')
y = to_categorical_np(y)

x_train, x_test, y_train, y_train = train_test_split(x,y,test_size=0.15, random_state=42)

class DeepNeuralNetwork:
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.params = self.initialization()
        
dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])

def initialization(self):
    input_layer=self.sizes[0]
    hidden_1=self.sizes[1]
    hidden_2=self.sizes[2]
    output_layer=self.sizes[3]

    params = {
        'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
        'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
        'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
    }

    return params

def forward_pass(self, x_train):
    params = self.params
    params['A0'] = x_train

    params['Z1'] = np.dot(params["W1"], params['A0'])
    params['A1'] = self.sigmoid(params['Z1'])

    params['Z2'] = np.dot(params["W2"], params['A1'])
    params['A2'] = self.sigmoid(params['Z2'])

    params['Z3'] = np.dot(params["W3"], params['A2'])
    params['A3'] = self.softmax(params['Z3'])

    return params['A3']

def backward_pass(self, y_train, output):
    params = self.params
    change_w = {}

    error = output - y_train
    change_w['W3'] = np.dot(error, params['A3'])

    error = np.multiply( np.dot(params['W3'].T, error), self.sigmoid(params['Z2'], derivative=True) )
    change_w['W2'] = np.dot(error, params['A2'])

    error = np.multiply( np.dot(params['W2'].T, error), self.sigmoid(params['Z1'], derivative=True) )
    change_w['W1'] = np.dot(error, params['A1'])

    return change_w

