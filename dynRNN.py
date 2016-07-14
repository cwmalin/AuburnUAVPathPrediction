import numpy as np
import pylab
file = open('newInput.txt','r')
counter=0

#Defines our activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
#Training parameters
inputSize = 10
hiddenDim = 25
alpha = .01*1.5
iterations = 500000
points=[]
np.random.seed(1)
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
evaluationY = Y[len(Y)//2:len(Y):1, 0:len(Y[0]):1]
Y = Y[0:len(Y)//2:1, 0:len(Y[0]):1]
#Parameters for our neural network
synapse0 = 2*np.random.random((2,hiddenDim)) - 1
synapse1 = 2*np.random.random((hiddenDim,2)) - 1
synapseH = 2*np.random.random((hiddenDim,hiddenDim)) - 1
l1 = np.zeros((len(X), hiddenDim), float)
synhDelta = np.zeros_like(synapseH, float)
syn0Delta = np.zeros_like(synapse0, float)
print (hiddenDim)
for iter in range(iterations):
    #Scale alpha down as we get more accurat
    if iter%100000==0:
        alpha/=1.5
        print(alpha)
    l1*=0
    layers1 = [np.zeros((len(X), hiddenDim), float)]
    i=0
    #computes our hiddenlayers
    while i<len(X[0]):        
        l1 = sigmoid(np.dot(X[0:len(X):1, i:i+2:1], synapse0)+np.dot(layers1[int(i/2)], synapseH))
        layers1.append(l1)
        i+=2
    layers2 = []
    #compute our output layers
    for j in range(len(layers1)):
        layers2.append(sigmoid(np.dot(layers1[j],synapse1)))
        
    i = len(layers2)-1
    l1Delta=np.zeros((len(X), hiddenDim))
    synHDelta=np.zeros((hiddenDim,hiddenDim))
    syn1Delta=np.zeros((hiddenDim,2))
    syn0Delta = np.zeros_like(synapse0, float)
    while i>0:
        synHDelta += layers1[i].T.dot(l1Delta)
        l2Error = Y[0:len(Y):1, 2*i-2:2*i:1]-layers2[i]
        l2Delta = -l2Error*layers2[i]*(1-layers2[i])
        syn1Delta += layers1[i].T.dot(l2Delta)
        l1Delta = (np.dot(l1Delta, synapseH.T)+np.dot(l2Delta,synapse1.T))*layers1[i]*(1-layers1[i])
        syn0Delta += np.dot(X[0:len(X):1, 2*i-2:2*i:1].T,l1Delta) 
        i-=1
    synapseH -= alpha*synHDelta
    synapse0 -= alpha*syn0Delta
    synapse1 -= alpha*syn1Delta
    if iter%10000==0:
            print(np.linalg.norm(Y[0:len(Y):1, 18:20:1]-layers2[-1]))
def activate(x, synapse_0, synapse_1, synapse_h):
    layers = [np.array(np.zeros((len(x), hiddenDim), float))]
    print(len(x))
    i=0
    while i<len(x[0]):
        l1 = sigmoid(np.dot(x[0:len(x):1, i:i+2:1], synapse_0)+np.dot(layers[int(i/2)], synapse_h))
        layers.append(l1)
        i+=2
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2