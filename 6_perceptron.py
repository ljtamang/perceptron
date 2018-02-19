
# coding: utf-8

# In[278]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import numpy as np
from numpy import genfromtxt


# In[279]:


def predict(x, weights):
    """ Predict the output label

        Parameters
        ----------
        :x: {array-like}, shape = [n_features]
                  Single Training record where n_features 
                  is the number of features in the training record.
        :weights: array-like, shape = [n_features+1]
                  Weights to be learned
        :return:  output : float 
                  Predicted traget label of the traing 

        """
    bias = weights[0] # First weight is bias term
    net_input = np.matmul(x, weights[1:]) + bias
    output = np.where(net_input>0.0, 1, -1)
    return output

def fit(x_train, y_train,learning_rate=0.1, no_of_epochs=500):
        """ Fit training data.

        Parameters
        ----------
        :x_train: {array-like}, shape = [n_samples, n_features]
                  Training recoreds, where n_samples
                  is the number of records and
                  n_features is the number of features of each record.
        :y_train: array-like, shape = [n_targets]
        :learning_rate: float, 
                  learning rate of the perceptron
        :no_of_epochs: float, 
                  no of epochs to learn the weight
        :return:  weight : array-like, shape = [n_weights]
                  weight learnded in each eopchs
        :return:  errors_percent : array-like, shape = [n_errors]
                  errors of each epochs

        """
        weights = np.random.rand(1+x_train.shape[1])
        errors_percent = [] # Stores error % for each epochs in percentage
        for epochs in range(no_of_epochs):
            errors = 0
            for x,target in zip(x_train, y_train): 
                output = predict(x, weights) # find predicted output
                weights_to_update = learning_rate*(target-output)
                weights[1:] += weights_to_update*x # update the weights excepts bias
                weights[0] += weights_to_update # update bias
                errors += int(weights_to_update != 0.0) # add 1 if output and target are not same
            erro_in_percent = float(100*float(errors)/float(len(x_train)))
            errors_percent.append(erro_in_percent) # add error percentage in each epochs
            
        return weights, errors_percent 


# In[280]:


def load_dataset1(csvFilename):
    """ Load data from csv.

        Parameter
        ----------
        :csvFilename: String
                      Name of the csv file which contains data
        :return     : dataset: {array-like}, shape = [n_rows, n_columns]
                      Records where n_rows is the number of records
                      and n_columns is the features in each record.
                
        """
    dataset = genfromtxt(csvFilename, delimiter=',', skip_header=1)
    no_columns = dataset.shape[1]
    x_train = dataset[:,:no_columns-1]
    y_train = dataset[:,no_columns-1]
    return x_train, y_train 


# In[281]:


def load_dataset2(csvFilename):
    """ Load data from csv.

        Parameter
        ----------
        :csvFilename: String
                      Name of the csv file which contains data
        :return     : dataset: {array-like}, shape = [n_rows, n_columns]
                      Records where n_rows is the number of records
                      and n_columns is the features in each record.
                
        """
    # find number of columns
    with open(csvFilename) as f:
        ncols = len(f.readline().split(','))
    # Read training data
    x_train = genfromtxt(csvFilename, delimiter=',',skip_header=1,usecols=range(0,ncols-1))
    # Read target data
    #for d-100 to convert NS->-1 and S-> 1
    convertfunc = lambda x: -1 if x==b'NS' else 1
    y_train=genfromtxt(csvFilename, delimiter=',',skip_header=1,usecols=range(ncols-1,ncols), converters={ncols-1: convertfunc})
    return x_train, y_train


# In[282]:


def display(dataset, learning_rate, errors_percent):
    """ Display the result.
    """
    print("Dataset name: %s" % (dataset))
    print("learning rate: %.3f" % (learning_rate))
    print("Error at 100 epochs: %.3f" % (errors_percent[99]))
    print("Error at 500 epochs %.3f" % (errors_percent[499]))
    print("*******************************" )


# In[284]:


# Train with d-10.csv  data
dataset = "d-10.csv"
no_of_epochs = 500
learning_rate = 0.01
x_train, y_train  = load_dataset1(dataset) # load data
_, errors_percent = fit(x_train, y_train , learning_rate, no_of_epochs) # fit the data to learn the weights
display(dataset, learning_rate, errors_percent)

learning_rate = 0.1
_, errors_percent = fit(x_train, y_train , learning_rate, no_of_epochs) 
display(dataset, learning_rate, errors_percent)

learning_rate = 0.2
_, errors_percent = fit(x_train, y_train , learning_rate, no_of_epochs) 
display(dataset, learning_rate, errors_percent)


# In[286]:


#  training with "d-100.csv"
dataset2 = "d-100.csv"
no_of_epochs = 500
learning_rate = 0.01
x_train, y_train  = load_dataset2(dataset2) # load data
_, errors_percent = fit(x_train, y_train , learning_rate, no_of_epochs) # fit the data to learn the weights
display(dataset2, learning_rate, errors_percent)

learning_rate = 0.1
_, errors_percent = fit(x_train, y_train , learning_rate, no_of_epochs) 
display(dataset2, learning_rate, errors_percent)

learning_rate = 0.2
_, errors_percent = fit(x_train, y_train , learning_rate, no_of_epochs) 
display(dataset2, learning_rate, errors_percent)

