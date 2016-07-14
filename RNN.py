import numpy as np
import pylab
file = open('input.txt','r')
inputPoints=[np.array([[-.1,-.1]])]
outputPoints=[]
counter=0

#Defines our activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
#Parses a list of comma separated points
for line in file:
    if line.isspace():
        counter+=1
        if counter%2==1:
            outputPoints.append(np.array([[-.1,-.1]]))
        else:
            inputPoints.append(np.array([[-.1,-.1]]))
    else:
        
        addTo=np.array([np.asarray(line.replace('/n','').split(','),float)])      
        if counter%2==1:    
            outputPoints[counter//2]=np.concatenate((outputPoints[counter//2],addTo),0)
        else:    
            inputPoints[counter//2]=np.concatenate((inputPoints[counter//2],addTo),0)
for i in range(len(inputPoints)):
    outputPoints[i]=np.delete(outputPoints[i],0,0)
    inputPoints[i]=np.delete(inputPoints[i],0,0)           
#Defines the training batches parameters
inputSize = 10
outputSize = 1

X = np.empty((len(inputPoints),2*inputSize), float)
xInputs = np.empty((len(inputPoints),inputSize),float)
yInputs = np.empty((len(inputPoints),inputSize),float)
Y = np.empty((len(outputPoints),2*outputSize),float)
xOutputs = np.empty((len(inputPoints),outputSize),float)
yOutputs = np.empty((len(inputPoints),outputSize),float)
#Makes the data more suitable for the network
for i in range(len(X)):
    for j in range(len(X[0])):
        X[i,j] = inputPoints[i][j//2,j%2]
        
    xInputs[i] = inputPoints[i][:,0]
    yInputs[i] = inputPoints[i][:,1]
for i in range(len(Y)):   
    for j in range(len(Y[0])):
        Y[i,j] = outputPoints[i][j//2,j%2]
    xOutputs[i] = outputPoints[i][:,0]
    yOutputs[i] = outputPoints[i][:,1]
#Neural Network Stuff
#scale down to avoid overflow
X/=1685
Y/=1685
evaluationX = X[len(X)//2:len(X):1, 0:len(X[0]):1]
X = X[0:len(X)//2:1, 0:len(X[0]):1]
evaluationY = Y[len(Y)//2:len(Y):1, 0:len(Y[0]):1]
Y = Y[0:len(Y)//2:1, 0:len(Y[0]):1]
#It's good to mess around with these parameters
alpha,hidden_dim = (.02,25)
np.random.seed(1)
print(alpha)
print(hidden_dim)
#Define the synapse parameters. Synapse_h are the synapses between hidden layers
synapse_0 = 2*np.random.random((2,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,2*outputSize)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1
#Keep track of our hidden layers for backprop
layers = [np.array(np.zeros((len(X), hidden_dim), float))]
l1 = np.zeros((len(X), hidden_dim), float)
synh_delta = np.zeros_like(synapse_h, float)
syn0_delta = np.zeros_like(synapse_0, float)
#Train
for iter in range(500000):
    #Scale alpha down as we get more accurate
    if iter%100000==0:
        alpha/=2
        print(alpha)
    l1*=0
    layers = [np.array(np.zeros((len(X), hidden_dim), float))]
    i=0
    #computes our layers
    while i<len(X[0]):
        l1 = sigmoid(np.dot(X[0:len(X):1, i:i+2:1], synapse_0)+np.dot(layers[int(i/2)], synapse_h))
        layers.append(l1)
        i+=2
    #All the backprop stuff
    l2 = sigmoid(np.dot(l1, synapse_1))
    l2_error = Y-l2
    l2_delta = -l2_error*l2*(1-l2)
    l_delta = np.dot(l2_delta,synapse_1.T)*l1*(1-l1)
    synh_delta *= 0
    syn0_delta = np.dot(X[0:len(X):1, len(X[0])-2:len(X[0]):1].T, l_delta)
    synapse_1 -= (alpha * l1.T.dot(l2_delta))
    i = len(layers)-2
    #Propagates through the hidden layers
    while i>0:
        synh_delta += (layers[i].T.dot(l_delta))
        l_delta = np.dot(l_delta, synapse_h.T)*layers[i]*(1-layers[i])
        
        #l_delta0 = np.dot(l_delta, synapse_0.T)*layers[i]*(1-layers[i])
        syn0_delta += np.dot(X[0:len(X):1, 2*i-2:2*i:1].T,l_delta)
        i-=1
    synapse_h -= alpha*synh_delta
    synapse_0 -= alpha*syn0_delta
    if iter%10000==0:
            print(np.linalg.norm(Y-l2))

def activate(x, synapse_0, synapse_1, synapse_h):
    layers = [np.array(np.zeros((len(x), hidden_dim), float))]
    i=0
    while i<len(x[0]):
        l1 = sigmoid(np.dot(x[0:len(x):1, i:i+2:1], synapse_0)+np.dot(layers[int(i/2)], synapse_h))
        layers.append(l1)
        i+=2
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2
    
def plotGuess(inputs):
    outputs=activate(inputs, synapse_0, synapse_1,synapse_h)
    xIn = np.concatenate((inputs[0, 0:10:1],outputs[0]),0)[0:22:2]
    yIn = np.concatenate((inputs[0,10:20:1], outputs[0]),0)[1:22:2]
    pylab.plot(1685*xIn, 1685*yIn)

def plot(xI,yI,xO,yO):
    xIn = np.concatenate((xI,xO),0)
    yIn = np.concatenate((yI,yO),0)
    pylab.plot(xIn,yIn)