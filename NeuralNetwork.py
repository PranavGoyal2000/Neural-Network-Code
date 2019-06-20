"""
Created on Wed Jun 15 17:55:35 2019

@author: pranav
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy

def Churn_Model(X_train, Y_train, X_test = None, Y_test = None, X_evaluate = None, 
                input_nodes, output_nodes, layers_list, functions_list, loss_function, 
                optimizer, epoch_number, batch_size):
    '''
    X_train is the training dataset (numpy array) of dimensions (m_train, n_X)
    Y_train is the outputs for X_train. (numpy array) of dimensions (m_train, n_Y)
    X_test (optional) can be used for testing your model. (numpy array) of dimensions (m_test, n_X)
    Y_test (optional) can be used for testing your model. (numpy array) of dimensions (m_test, n_Y)
    X_evaluate (optional) returns the expected labels according to the trained model. (m_eval, n_X)
    input_nodes is n_X
    output_nodes is n_Y
    layers_list is list of number of nodes in the hidden layers followed by n_Y
    functions_list is a list of strings that denote the functions used to compute each hidden layer and 
                   output layer
    loss_function is a string denoting the loss function definition
    optimizer is a string that represents the algorithm optimizing the cost function
    epoch_number is the number of epochs you want to run the code
    batch_size is the size of batch you want to use before running the optimizer
    '''
    assert(len(layer_list) == len(functions_list))
    assert(output_nodes == layer_list[(len(layer_list) - 1)])
    assert(X_train.shape[0] == Y_train.shape[0])
    assert(X_test.shape[0] == Y_test.shape[0] or (X_test == None and Y_test == None))
    assert(type(loss_function) == str)
    assert(type(optimizer) == str)
    
    numpy.random.seed(0)
    model = Sequential()
    model.add(Dense(layers_list[0], input_dim = input_nodes, activation = functions_list[0]))
    
    for i in range(len(layers_list) - 1):
        model.add(Dense(layers_list[i + 1], activation = functions_list[i + 1]))
    
    model.compile(loss = function_loss, optimizer = optimizer, metrics = ['accuracy'])
    model.fit(X_train, Y_train, epochs = epoch_number, batch_size = batch_size)
    scores1 = model.evaluate(X_train, Y_train)
    
    if X_test == None:
        if X_evaluate == None:
            return scores1[1]
        else:
            predictions = model.predict(X_evaluate)
            return (scores1[1], predictions)
    else:
        scores2 = model.evaluate(X_test, Y_test)
        if X_evaluate == None:
            return (scores1[1], scores2[1])
        else:
            predictions = model.predict(X_evaluate)
            return (scores1[1], scores2[1], predictions)
        
'''
Churn_Model(_, _, _, _, _, 8, 1, [12, 8, 1], ['relu', 'relu', 'sigmoid'], 'binary_crossentropy', 'Adam', 150, 10)
'''