import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM
from keras.models import Sequential

def Sequence_Model(X_train, Y_train, X_test, Y_test, X_evaluate, n_a, f_a_to_Y, loss_function, 
                   optimizer, epoch_number, batch_size):
    '''
    X_train is the training dataset (numpy array) of dimensions (m_train, sequence_length, no_parameters)
    Y_train is the outputs for X_train. (numpy array) of dimensions (m_train)
    X_test (optional) can be used for testing your model. (numpy array) of dimensions (m_test, sequence_length, no_parameters)
    Y_test (optional) can be used for testing your model. (numpy array) of dimensions (m_test)
    X_evaluate (optional) returns the expected labels according to the trained model. (m_eval, sequence_length, no_parameters)
    n_a is the length of memory list a created by the LSTM
    f_a_to_Y is a string denoting the function that is used to obtain Y from a.
    loss_function is a string denoting the loss function definition
    optimizer is a string that represents the algorithm optimizing the cost function
    epoch_number is the number of epochs you want to run the code
    batch_size is the size of batch you want to use before running the optimizer
    '''
    assert(X_train.shape[0] == Y_train.shape[0])
    assert((str(type(X_test)) == "<class 'NoneType'>" and str(type(Y_test)) == "<class 'NoneType'>") or X_test.shape[0] == Y_test.shape[0])
    assert(type(loss_function) == str)
    assert(type(optimizer) == str)
    np.random.seed(0)
    
    model = Sequential()
    model.add(LSTM(n_a))
    model.add(Dense(1, activation = f_a_to_Y))
    model.compile(loss = loss_function, optimizer = optimizer, metrics = ['accuracy'])
    model.fit(X_train, Y_train, epochs = epoch_number, batch_size = batch_size)
    
    scores1 = model.evaluate(X_train, Y_train, verbose = 0)
    
    if str(type(X_test) == "<class 'NoneType'>":
        if str(type(X_evaluate)) == "<class 'NoneType'>":
            return scores1[1]
        else:
            predictions = model.predict(X_evaluate)
            return (scores1[1], predictions)
    else:
        scores2 = model.evaluate(X_test, Y_test, verbose = 0)
        if str(type(X_evaluate) == "<class 'NoneType'>":
            return (scores1[1], scores2[1])
        else:
            predictions = model.predict(X_evaluate)
            return (scores1[1], scores2[1], predictions)
    
'''
Sequence_Model(_, _, _, _, _, 100, 'sigmoid', 'binary_crossentropy', 'Adam', 150, 10)
'''
