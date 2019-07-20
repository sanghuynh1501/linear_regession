import tensorflow as tf
import pandas as pd
import numpy as np

def read_csv(data_path):
    data = pd.read_csv(data_path)
    return data.values

def prediction(x, weight, bias):
    return weight*x + bias # our predicted (learned) m and c, expression is like y = m*x + c

def loss(x, y, weights, biases): 
    error = prediction(x, weights, biases) - y # how 'wrong' our predicted (learned) y is
    squared_error = tf.square(error)
    return tf.reduce_mean(input_tensor=squared_error) # overall mean of squared error, scalar value.

def grad_weight(x, y, weights, biases):
    with tf.GradientTape() as tape:
        tape.watch(weights)
        loss_ = loss(x, y, weights, biases)
    return tape.gradient(loss_, weights) # direction and value of the gradient of our weights and biases

def grad_bias(x, y, weights, biases):
    with tf.GradientTape() as tape:
        tape.watch(biases)
        loss_ = loss(x, y, weights, biases)
    return tape.gradient(loss_, biases) # direction and value of the gradient of our weights and biases

streets = ['Pave', 'Grvl']
neighborhoods = [
    'CollgCr', 'Veenker', 'Crawfor',
    'NoRidge', 'Mitchel', 'Somerst',
    'NWAmes',  'OldTown', 'BrkSide',
    'Sawyer',  'NridgHt', 'NAmes',
    'SawyerW', 'IDOTRR',  'MeadowV',
    'Edwards', 'Timber',  'Gilbert',
    'StoneBr', 'ClearCr', 'NPkVill',
    'Blmngtn', 'BrDale',  'SWISU',
    'Blueste'
]

data = read_csv("data/train.csv")
x_train = []
y_train = []
for i in range(len(data)):
    x = []
    y = []
    print('data[i] ', data[i])
    x.append(float(data[i][1]) / 9986)
    if data[i][2] in streets:
        x.append(float(streets.index(data[i][2])) / 2)
    else:
        x.append(1)
    x.append(float(neighborhoods.index(data[i][3])) / len(neighborhoods))
    x.append(float(data[i][4]) / 2009)
    x.append(float(data[i][5]) / 2010)
    x_train.append(x)
    y.append(float(data[i][6]) / 99900)
    y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)
print("np.random.randn() ", np.random.randn())
W = tf.Variable(np.random.randn()) # initial, random, value for predicted weight (m)
B = tf.Variable(np.random.randn()) # initial, random, value for predicted bias (c)
learning_rate = 0.01
print("Initial loss: {:.3f}".format(loss(x, y, W, B)))
for step in range(3000): #iterate for each training step
    deltaW = grad_weight(x_train, y_train, W, B)
    deltaB = grad_bias(x_train, y_train, W, B) # direction(sign) and value of the gradients of our loss 
    change_W = deltaW * learning_rate # adjustment amount for weight
    change_B = deltaB * learning_rate # adjustment amount for bias
    W = W - change_W # subract change_W from W
    B = B - change_B # subract change_B from B
    print("Loss at step {:02d}: {:.6f}".format(step, loss(x, y, W, B)))