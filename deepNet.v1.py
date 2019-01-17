import numpy as np
import matplotlib.pyplot as plt
from nn_utils import *

def initial_parameters(layers):

    parameters ={}
    L = len(layers)

    for l in range(1,L):
        parameters['W'+ str(l)] = np.random.rand(layers[l],layers[l-1])
        parameters['b'+ str(l)] = np.random.rand(layers[l],1)

    return parameters

def single_layer_forward(A,l,parameters):

    W = parameters['W'+str(l)]
    b = parameters['b'+str(l)]
    Z = np.dot(W,A) + b

    A_next = sigmoid(Z)

    return A_next, Z

def forward_pass(X,layers,parameters):

    L = len(layers)

    caches = []
    caches.append([-1])
    A_prev = X
    for l in range(1,L):
        A_next,Z = single_layer_forward(A_prev,l,parameters)
        cache = {'A'+str(l-1):A_prev,'Z'+str(l):Z,'A'+str(l):A_next}
        caches.append(cache)
        A_prev = A_next

    return A_next, caches


def single_layer_backward(da,l,parameters,caches):

    Al = caches[l]['A'+str(l)]

    dz_l = np.multiply( Al , (1-Al))
    dA_l_1 = np.dot( parameters['W'+str(l)].T, dz_l)

    dw_l = np.dot( dz_l, caches[l]['A'+str(l-1)].T )
    db_l = np.sum(dz_l,1).reshape(-1,1)

    return dA_l_1, dw_l, db_l

def backward_pass(dA_L,layers,parameters,caches,lr):

    L = len(layers)

    dA_next = dA_L
    for l in range(L-1,0,-1):

        dA_prev, dW, db = single_layer_backward(dA_next,l,parameters,caches)
        dA_next = dA_prev

        parameters['W'+str(l)] = parameters['W'+str(l)]-lr*dW
        parameters['b' + str(l)] = parameters['b' + str(l)] - lr * db

    return parameters

#---------------------Main---------------------------
    

layers = [5,10,2,1]
parameters = initial_parameters(layers)

X = np.random.rand(5,100)
Y = np.random.rand(1,100)
# A, Z = single_layer_forward(X,1,parameters)
AL , caches = forward_pass(X,layers,parameters)
dA_L = np.random.rand(1,100)
lr = .01
# dA, dW, db = single_layer_backward(da,2,parameters,caches)


backward_pass(dA_L,layers,parameters,caches,lr)






#################### add optimizer and compute L(Y^,Y) as cost function ###########


def optimizer_fullbatch_GD(epoch=5,num_iterations=10):
    
    for e in range(epoch):
        
        for i in range(num_iterations):
            forward_pass(X,layers,parameters)
            backward_parameters = backward_pass(dA_L,layers,parameters,caches,lr)

            
            weight = []
            bias = []
            A_prev = X
            
            for l in range(1,int(len(backward_parameters)/2)+1,1):
                weight = backward_parameters['W'+str(l)]
                bias = backward_parameters['b'+str(l)]
                A_prev = sigmoid(np.dot(weight, A_prev) + bias)
            
            cost = (- 1 / X.shape[1]) * np.sum(Y * np.log(A_prev) + (1 - Y) * np.log((1 - A_prev)))
            print("epoch : %s itr : %s cost is : %f"%(e,i,cost))
    

optimizer_fullbatch_GD(epoch=5,num_iterations=2)



def optimizer_minibatch_GD(epoch=5, num_iterations=10, batch=3 , lr = .01):
    for e in range(epoch):

        for itr in range(num_iterations):
            batches_x = np.array_split(X, 3)
            batches_y = np.array_split(Y, 3)
            layers = [batches_x[0].shape[1],10,2,1]
            cost = 0 
            parameters = initial_parameters(layers)

            for i in range(len(batches_x)):
                AL , caches = forward_pass(batches_x[i].T, layers, parameters)
                dA_L = np.random.rand(1,len(batches_x[i]))
                backward_parameters = backward_pass(dA_L, layers, parameters, caches, lr)

                weight = []
                bias = []
                A_prev = batches_x[i].T

                for l in range(1, int(len(backward_parameters) / 2) + 1, 1):
                    weight = backward_parameters['W' + str(l)]
                    bias = backward_parameters['b' + str(l)]
                    A_prev = sigmoid(np.dot(weight, A_prev) + bias)

                cost += (- 1 /len(batches_x)) * np.sum(batches_y[i].reshape(-1,1) * np.log(A_prev) + (1 - batches_y[i].reshape(-1,1)) * np.log((1 - A_prev)))
            print("epoch : %s itr : %s cost is : %f"%(e,itr,cost))


optimizer_minibatch_GD(epoch=2, num_iterations=10, batch=3 , lr = .01)


temp = 1