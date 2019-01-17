#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author : mehdi shamkhani 

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

tf.enable_eager_execution()
tf.executing_eagerly() 


mnist = tf.keras.datasets.mnist

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, tl), (test_images, test_labels) = fashion_mnist.load_data()


train_labels= [0 if i != 2 else 1 for i in train_labels]
test_labels= [0 if i != 2 else 1 for i in test_labels]

train_labels= np.asarray(train_labels)
test_labels= np.asarray(test_labels)

train_labels = train_labels
x_flatten = train_images.reshape(train_images.shape[0],-1)
xtest_flatten = test_images.reshape(test_images.shape[0],-1)


x_flatten = x_flatten/255.
xtest_flatten =xtest_flatten/255.

def sigmoid(z):
  
    s = 1 / (1 + np.exp(-z))
   
    return s

  
def initialize_params(dim):

    w = np.zeros(shape=(dim, 1))
    b = 1

    return w, b
  

def propagation(w, b, X, Y):
   
    
    
    train_size = X.shape[1]
    
  
    A = sigmoid(np.dot(w.T, X) + b)
   
    
    cost = (- 1 / train_size) * np.sum(Y * np.log(A) + (1 - Y) * np.log((1 - A)))
    
  
    dw = (1 / train_size) * np.dot(X, (A - Y).T)
    db = (1 / train_size) * np.sum(A - Y)
   
    
    gradients_dict = {"dw": dw,
             "db": db}
    
    return gradients_dict, np.squeeze(cost)
  
  
def optimizing(w, b, X, Y, num_iterations, learning_rate):
   
    costs = []
    
    for i in range(num_iterations):
        
     
        gradients_dict, current_cost = propagation(w, b, X, Y)
      
        dw = gradients_dict["dw"]
        db = gradients_dict["db"]
               
        w = w - learning_rate * dw 
        b = b - learning_rate * db

    
        costs.append(current_cost)
        
        
    model_params = {"weights": w,
              "bias": b}    

    
    return model_params
  
  
  
def predict(w_, b, X):
  
    
    m = X.shape[0]
    label_pred = np.zeros((1, m))
    A = sigmoid(np.dot(w_.T, X) + b)
    

    for i in range(A.shape[1]):
        label_pred[0][i] =  1 if A[0][i] > 0.5   else 0
    
    
    return label_pred

  


  
  
  
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]




index = 10
plt.imshow(train_images[index])

w, b = initialize_params(x_flatten.shape[0])

model_params = optimizing (w, b ,x_flatten , train_labels.reshape(-1,1) , num_iterations=20000, learning_rate = 0.01)

predicted_label = predict(model_params["weights"], b, xtest_flatten)

accu = 100 - np.mean(np.abs(predicted_label - test_labels)) * 100

print("accuracy : %f "%accu)