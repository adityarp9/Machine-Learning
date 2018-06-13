from __future__ import print_function
import sys
import os

import scipy.io
import numpy as np

import bokeh.plotting as bk
from scipy.interpolate import interp1d

# Data and its processing
def prepareData(dataDict):
    y_test = dataDict['ytest']
    y_val = dataDict['yval']
    y = dataDict['y']
   
    X_test = dataDict['Xtest']
    X_val = dataDict['Xval']
    X = dataDict['X']

    return [X, X_val, X_test, y, y_val, y_test] 

def polyFeatures(x, p): 
    for deg in range(2,p+1):
        x = np.concatenate((x, np.expand_dims(x[:, 1]**deg, axis=1)), axis=1)
    return x

def featureNormalize(x, mean=None, stdev=None):
    if mean is None and stdev is None:
        mean = np.mean(x[:, 1:], axis=0)
        stdev = np.std(x[:, 1:], axis=0)
    x[:, 1:] -= mean
    x[:, 1:] /= stdev 
    return x, mean, stdev   
 

# Model
def model(inputs, params):
    return np.dot(inputs, params)


# Cost functions
def costFunction(out, y):
    cost = 0.5 * (np.mean((out - y) ** 2))
    return cost 

def regCostFunction(m, cost, theta, lambdaa):
 
    cost_reg = np.sum(theta ** 2) * 0.5 * (lambdaa / m)
    cost += cost_reg
    return cost


# Gradients
def grad_cost(inputs, outputs, y, params, lambdaa):
    grad = np.zeros(params.shape)
    factor = 1.0 / inputs.shape[0]
    loss = outputs - y 
    grad[0] = (np.sum(np.dot(inputs[:, 0].T, (loss))))
    for i in range(1, inputs.shape[1]):
        grad[i] = (np.sum(np.dot(inputs[:, i].T, (loss)))) + (lambdaa/ inputs.shape[0]) * params[i]
    grad *= factor
    return grad


# Training
def train(inputs, params, hypers, y):
    out = model(inputs, params) 
    cost = costFunction(out, y)
    #print(cost) 
    cost = regCostFunction(y.shape[0], cost, params, hypers['lambda'])
    #print(cost)
    grads = grad_cost(inputs, out, y, params, hypers['lambda'])
    #print(grads)
    #sys.exit(1)
    #print("Grads\n{}".format(grads)) 
    updateParams(params, grads, hypers['lr'])
    print("Train error: {}".format(cost))
    return out


# Parameters
def initParams(num_features, kind="zeros"):
    if kind == "zeros":
        return np.zeros((num_features, 1))
    if kind == "ones":
        return np.ones((num_features, 1))
    if kind == "random":
        return np.random.rand(num_features, 1)

def updateParams(params, grads, lr):
    params += -lr * grads


def main():
    if sys.argv == 1:
        sys.exit("Invalid clargs")
    filename = sys.argv[1]

    if not os.path.exists(filename):
        sys.exit("Data file not found.")
    mat = scipy.io.loadmat(filename)
    # Format: data -> [X, X_val, X_test, y, y_val, y_test]
    data = prepareData(mat)

    X_train = np.concatenate((np.ones(data[0].shape), data[0]), axis=1) # Training input
    X_val = np.concatenate((np.ones(data[1].shape), data[1]), axis=1) # Validation input
    X_test = np.concatenate((np.ones(data[2].shape), data[2]), axis=1) # Testing input
 
    theta = initParams(X_train.shape[1], kind="ones") # Parameters initialize

    # Rehersal
    out = model(X_train, theta) # output of model
    J = costFunction(out, data[3])
    #print("Cost for theta =\n{} is {}".format(theta, J))
    reg_str = 1.0
    Jreg = regCostFunction(data[3].shape[0], J, theta, reg_str)
    #print("Regularized cost for theta =\n{} is {}".format(theta, Jreg))
    grad_cost(X_train, out, data[3], theta, reg_str)

    # Linear regression
    # Hypers
    hypers = {'lr':0.08, 'lambda':reg_str, 'max_epochs':50}
    
    # Normalize dataset inputs
    xTr_norm, mean, stdev = featureNormalize(X_train)
    xVa_norm, _, _ = featureNormalize(X_val, mean, stdev)
    XTe_norm, _, _ = featureNormalize(X_test, mean, stdev)

    for epoch in range(hypers['max_epochs']): 
        train(xTr_norm, theta, hypers, data[3])
        #print("Parameters: \n{}".format(theta)) 
    '''
    # Plotting linear regressor
    bk.output_file("line.html")
    hypot_fig = bk.figure()
    hypot_fig.circle(np.squeeze(xTr_norm[:, 1:]), np.squeeze(data[3]), color="red", legend="Input train data") 
    hypot_fig.line(np.squeeze(xTr_norm[:, 1:]), np.squeeze(model(xTr_norm, theta), axis=1), color="green", legend="Hypothesis")
    bk.show(hypot_fig)
    '''
    # Learning curves
    theta = initParams(xTr_norm.shape[1])
    error_train, error_val = [], []  
    for size in range(X_train.shape[0]):
        print("\nTraining set size: {}".format(xTr_norm[0:size+1].shape))
        for epoch in range(hypers['max_epochs']): 
            out = train(xTr_norm[0:size+1], theta, hypers, data[3][0:size+1])
        error_train.append(costFunction(out, data[3][0:size+1]))
        error_val.append(costFunction(model(xVa_norm, theta), data[4]))
    '''
    # Plotting learning curves
    bk.output_file("line1.html")
    error_fig = bk.figure()
    error_fig.line(range(len(error_train)), error_train, color="red", legend="Training error") 
    error_fig.line(range(len(error_val)), error_val, color="green", legend="Cross-validation error")
    bk.show(error_fig)
    '''

    # Polynomial regression
    print("\nPolynomial regression")
    p = 8
    xTr_norm_p = polyFeatures(xTr_norm, p)
    xVa_norm_p = polyFeatures(xVa_norm, p)
    hypers['max_epochs'] = 500
    hypers['lr'] = 0.009
    hypers['lambda'] = 1.0
    theta = initParams(xTr_norm_p.shape[1])
    for epoch in range(hypers['max_epochs']):
        #print("Parameters:\n{}".format(theta))
        train(xTr_norm_p, theta, hypers, data[3])

    # Plotting poly-regressor fit
    bk.output_file("line2.html")
    x_plot = np.squeeze(xTr_norm[:, 1:])
    y_plot = np.squeeze(np.squeeze(data[3]))
    x_min = np.min(x_plot)
    x_max = np.max(x_plot)
    f = interp1d(x_plot, y_plot, kind="quadratic")
    x_interpol = np.linspace(x_min, x_max, 1000)
    y_interpol = f(x_interpol)
 
    poly_hypot_fig = bk.figure(y_range=(-20, 50), x_range=(-5, 5))
    poly_hypot_fig.circle(np.squeeze(xTr_norm[:, 1:]), np.squeeze(data[3]), color="red", legend="Input train data with poly features") 
    poly_hypot_fig.line(x_interpol, y_interpol, color="blue", legend="Hypothesis", line_dash="dashed", line_width=2)
    bk.show(poly_hypot_fig) 


if __name__ == "__main__":
    main()
