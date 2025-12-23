import random
import numpy as np
import time 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = np.array(x, dtype='float32') / 255  # Convert to numpy array
y = np.array(y, dtype=int)  # Convert to numpy array
y = to_categorical_np(y)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15, random_state=42)

class DeepNeuralNetwork:
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate
        self.params = self.initialization() 
        
    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)
    
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
        params['A0'] = x_train.reshape(-1, 1) if x_train.ndim == 1 else x_train
                
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
    
    def update_network_parameters(self, changes_to_w):
    
        for key, value in changes_to_w.items():
            for w_arr in self.params[key]:
                w_arr -= self.l_rate * value
    
    def compute_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == y)

        summed = sum(pred for pred in predictions) / 100.0
        return np.average(summed)
    
    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)

            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
                iteration+1, time.time() - start_time, accuracy
            ))
    

dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_test, y_test)
