# -*- coding: utf-8 -*-
# This Neural Network is designed to predict the future position of a plane.
# The data is assumed to come from a plane's past 200 seconds of flying in a 500x500 grid
# The sampling rate of the input does not matter as we attempt to transform it into a Bernstein polynomial
# Obviously lower sampling rates mean we don't have as good an idea of what the output should be
# Higher sampling rates tend to be better, but at a certain point the bernstein polynomials can't model them accurately
# It will be important to find the right degree of polynomial because it is essentially normalizing our data
# It transforms the vector space of possible inputs into the vector space of nth degree polynomials
# A higher degree polynomial will be more accurate, but it also will increase training time
# The degree necessarily needs to be less than the number of points it is meant to fit because we are trying to limit the inputs to our network
# Mathematically this is a problem for most functions, but hopefully due to the constraints of flight it won't be as limiting

# The data to train this network is read from a file of CSV's
# 1: x input as a function of time. 2: x output as a function of time.
# 3: y input as a function of time. 3: y output as a function of time

# We read in the training data to a list

import numpy as np
file = open('input.txt','r')
inputPoints=[np.array([[-1,-1]])]
outputPoints=[]
points=[]
counter=0
inputSize=10
#Parses a list of comma separated points
for line in file:
    points+=line.replace('\n','').split(',')
    counter+=1
X = np.empty((len(points)//2-inputSize,2*inputSize),float)
Y = np.empty((len(points)//2-inputSize,2*inputSize),float)
i=0
for i in range(len(X)):
    X[i] = points[2*i:2*i+2*inputSize:1]
    Y[i] = points[2*i+2:2*i+2*inputSize+2:1]
X/=1685
Y/=1685
evaluationX = X[len(X)//2:len(X):1, 0:len(X[0]):1]
X = X[0:len(X)//2:1, 0:len(X[0]):1]
evaluationY = Y[len(Y)//2:len(Y):1, len(Y[0])-2:len(Y[0]):1]
Y = Y[0:len(Y)//2:1, len(Y[0])-2:len(Y[0]):1]        

#Neural Network Stuff
alpha,hidden_dim = (.01*1.5,35)
np.random.seed(1)
print(alpha)
synapse_0 = 2*np.random.random((2*inputSize,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,2)) - 1
lastValue = 999
noImprov=0
for j in range(500000):
    if j%100000==0:
        alpha/=1.5
        print(alpha)
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
    layer_2_delta = (layer_2 - Y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))
    value=np.linalg.norm(Y-layer_2)
    if j%10000==0:
        print (np.linalg.norm(Y-layer_2))
    lastValue=value
layer_2x = 1685*layer_2[0:len(layer_2):1, 0:1:1]
layer_2y = 1685*layer_2[0:len(layer_2):1, 1:2:1]

def activate(inputs):
    return 1/(1+np.exp(-(np.dot(1/(1+np.exp(-(np.dot(inputs,synapse_0)))),synapse_1))))