import pandas as pd
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Lambda, Concatenate
from keras.optimizers import Adam

def Churn_Sequence_Dense(X_train, Y_train, X_test, Y_test, X_evaluate, no_of_sequences, n_a_list,  f_a_to_Y_list, 
                         layers_list, functions_list, loss_function, optimizer, epoch_number, batch_size):
    '''
    X_train is the training dataset (numpy array) of dimensions (m_train, no_sequences, sequence_length, no_parameters)
    Y_train is the outputs for X_train. (numpy array) of dimensions (m_train)
    X_test (optional) can be used for testing your model. (numpy array) of dimensions (m_test, no_sequences, sequence_length, no_parameters)
    Y_test (optional) can be used for testing your model. (numpy array) of dimensions (m_test)
    X_evaluate (optional) returns the expected labels according to the trained model. (m_eval, no_sequences, sequence_length, no_parameters)
    no_of_sequences is the number of input sequences
    n_a_list is the list of lengths of memory list a of LSTM created for the sequences
    f_a_to_Y_list is a list of strings denoting the function that is used to obtain Y from a for all sequences
    layers_list is list of number of nodes in the hidden layers followed by n_Y
    functions_list is a list of strings that denote the functions used to compute each hidden layer and 
                   output layer
    loss_function is a string denoting the loss function definition
    optimizer is a string that represents the algorithm optimizing the cost function
    epoch_number is the number of epochs you want to run the code
    batch_size is the size of batch you want to use before running the optimizer
    '''
    assert(X_train.shape[0] == Y_train.shape[0])
    assert((str(type(X_test)) == "<class 'NoneType'>" and str(type(Y_test)) == "<class 'NoneType'>") or X_test.shape[0] == Y_test.shape[0])
    assert(type(loss_function) == str)
    assert(type(optimizer) == str)
    assert(no_of_sequences == X_train.shape[1])
    assert(no_of_sequences == len(n_a_list) and no_of_sequences == len(f_a_to_Y_list))
    np.random.seed(0)
    
    X = Input(shape = X_train.shape[1:])
    internal_outputs = []
    
    print(X.shape)
    
    for i in range(X.shape[1]):
        x = Lambda(lambda x: x[:,i,:,:])(X)
        x = LSTM(n_a_list[i])(x)
        x = Dense(1, activation = f_a_to_Y_list[i])(x)
        internal_outputs.append(x)
    
    second = Concatenate(axis = -1)(internal_outputs)
    print(internal_outputs, second.shape)
    
    output = Dense(layers_list[0], input_dim = no_of_sequences, activation = functions_list[0])(second)
    for i in range(len(layers_list) - 1):
        output = Dense(layers_list[i + 1], activation = functions_list[i + 1])(output)
    
    model = Model(inputs = X, outputs = output)
    model.compile(optimizer = optimizer, loss = loss_function, metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs = epoch_number)
    scores1 = model.evaluate(X_train, Y_train)
    
    if str(type(X_test)) == "<class 'NoneType'>":
        if str(type(X_evaluate)) == "<class 'NoneType'>":
            return scores1[1]
        else:
            predictions = model.predict(X_evaluate)
            return (scores1[1], predictions)
    else:
        scores2 = model.evaluate(X_test, Y_test)
        if str(type(X_evaluate)) == "<class 'NoneType'>":
            return (scores1[1], scores2[1])
        else:
            predictions = model.predict(X_evaluate)
            return (scores1[1], scores2[1], predictions)    
        


print(Churn_Sequence_Dense(np.array([[[[1,2,3], [1,2,3]], [[1,2,3],[1,2,3]]], 
                                     [[[1,2,3], [1,2,3]], [[1,2,3],[1,2,3]]]]),
                           np.array([0,1]), None, None, None, 2, [10, 10], ['sigmoid', 'sigmoid'], [3,1], ['relu', 'sigmoid'],
                           'binary_crossentropy', 'Adam', 50, 2))
