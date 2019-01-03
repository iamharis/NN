import numpy as np
from numpy import exp, array, random

def sigmoid(x):#The function returns the output of sigmoid function i.e 0 or 1
        return 1 / (1 + exp(-x))

def sigmoid_deri(x):#Dericative of sigmoid function, used in the calulation of the weights
        return x * (1 - x)

def train(trainx, trainY,No_of_iter,b): # Function that trains the NN and updates the weight
        for iteration in range(No_of_iter):
            q = np.dot(trainx,b)
            output = sigmoid(q)
            error = trainY - output
            adjustment = np.dot(trainx.T, error * sigmoid_deri(output))
            b += adjustment
        return b   
def think(inputs,b):
        a = np.dot(inputs, b)
        return sigmoid(a)

random.seed()
wt = random.random((3, 1)) # initializing random weights 
print ("Initialised random weights are:\n",wt,'\n')
trainx = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
trainY = array([[0], [1], [1], [0]])
wt_new = train(trainx, trainY, 100000, wt)
print ("Waits after training are: \n", wt_new,'\n')
print ("For the input: [1, 0, 0] we get the output as: \n")
m = array([1,0,0])
print(think(m,wt_new)) 