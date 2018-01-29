# -*- coding:utf-8 -*-
#/usr/bin/python

import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, auc
import time

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def sigmoid( z ):
    return  1.0/(1+ np.exp(-z))


def train_mgd( x, y, iter_num, learning_rate, eplise):
    # Mini-batch Gradient Descent
    # 每一步都用到全量的数据进行梯度下降的计算
    # learning_rate会在每次迭代中找寻最优化的
    nrow, ncol = x.shape
    x = np.hstack( (np.array([1.0 for i in xrange(nrow)]).reshape(nrow,1),x) )
    nrow, ncol = x.shape

    theta = np.ones((ncol, 1))
    costJ = []
    eplises = []
    e = 0.01
    alpha = learning_rate

    step = nrow/10

    for k in range( iter_num ):
        for i in range(step-1):
            z = np.dot( x[10*i:10*(i+1)], theta )
            h = sigmoid( z )
            J = ( np.sum(y[10*i:10*(i+1)] - h)**2 )/( 2*nrow )
            costJ.append( J )

            gradient = -np.dot( np.transpose(x[10*i:10*(i+1)] ), y[10*i:10*(i+1)] -h ) / nrow
            ep = sum( np.fabs(gradient) )
            eplises.append(ep)
            if ep < eplise:
                return theta, costJ, eplises

            #step = 0.001
            #a, b = get_ab_simple( x, y, theta, alpha, step, gradient, nrow )
            theta = theta + alpha * gradient
    return theta, costJ, eplises




def get_cost( x, y, theta, gradient, nrow, alpha):
    theta = theta - alpha*gradient
    z = np.dot( x, theta )
    h = sigmoid( z )
    return ( np.sum( (y-h)**2 ) )/(2*nrow)


def get_ab_simple( x, y, theta, alpha, step, gradient, nrow ):

    loop = 1

    J1 = get_cost( x, y, theta, gradient, nrow, alpha)
    alpha = alpha + step

    J2 = get_cost( x, y, theta, gradient, nrow, alpha)

    while ( loop > 0):
        if J1 > J2:
            step = 2 * step
        elif J1 <= J2:
            step = -2 * step
            alpha = alpha - step
            J2 = J1

        alpha_new = alpha + step
        J3 = get_cost( x, y, theta, gradient, nrow, alpha_new)
        if J3 > J2:
            a = min( alpha, alpha_new )
            b = max( alpha, alpha_new )
            return a, b
        else:
            alpha = alpha_new
            J1 = J2
            J2 = J3
        loop += 1

def predction( x, model):

    x_pred = []
    nrow, ncol = x.shape
    x_test = np.hstack( (np.array([1.0 for i in xrange(nrow)]).reshape(nrow,1),x) )

    for line in x_test:

        z = np.dot( model,line ) 
        h = 1.0/(1+ np.exp(-z))

        if h >= 0.5:
            x_pred.append( 1 )
        else:
            x_pred.append( 0 )


    return x_pred

def train_sgd( x, y, iter_num, learning_rate, eplise ):
    # stochastic gradient descent
    nrow, ncol = x.shape
    # 在前面增加一个常数项, x0
    x = np.hstack( (np.array([1.0 for i in xrange(nrow)]).reshape(nrow,1),x ) )
    nrow, ncol = x.shape
    w = np.zeros((1,ncol))

    costJ = []
    alpha = learning_rate
    lamda= 0.1
    k = 0
    i = 0
    for k in range( iter_num ):
        for i in range(nrow):
            x_sample = x[i]
            z = np.dot( w,x_sample)
            h = 1.0/(1+ np.exp(-z))
            gradient = ( x_sample.reshape(ncol,1) * (y[i]-h) - alpha*lamda*w.T ) / nrow

            w += alpha*gradient.T
            i += 1

        #J =  (np.sum((y-sigmoid(np.dot(x,w)).reshape(nrow,1))**2))/(2*nrow) + lamda*np.sum( w**2 )/(2*nrow)
        #costJ.append( J )
        if np.sum( np.fabs(gradient) ) <= eplise:
            return w, costJ
        k += 1

    return w, costJ


def train_bgd( x, y, iter_num, learning_rate, eplise):
    # Batch Gradient Descent
    # 每一步都用到全量的数据进行梯度下降的计算
    # learning_rate会在每次迭代中找寻最优化的
    nrow, ncol = x.shape
    x = np.hstack( (np.array([1.0 for i in xrange(nrow)]).reshape(nrow,1),x) )
    nrow, ncol = x.shape
    w = np.zeros((ncol,1))

    costJ = []
    eplises = []
    e = 0.01
    alpha = learning_rate
    lamda = 0.1
    for k in range( iter_num ):

        z = np.dot( x, w )
        h = sigmoid( z )

        #J = ( np.sum(y - h)**2 )/( 2*nrow )  + lamda*np.sum( w**2 )/(2*nrow)
        #costJ.append( J )

        gradient = -np.dot( x.T, (y.reshape(nrow,1)-h) ) / nrow - alpha*lamda*w / nrow
        ep = sum( np.fabs(gradient) )
        eplises.append(ep)
        if ep < eplise:
            return w.T, costJ, eplises

        #step = 0.001
        #a, b = get_ab_simple( x, y, w, alpha, step, gradient, nrow )
        w = w + alpha * gradient

    return w.T, costJ, eplises


if __name__== '__main__':

    print 'Start'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]


    train_features, test_features, train_labels, test_labels = train_test_split( imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print 'data processing cost: ',time_2 - time_1


    time_1 = time.time()

    iter_num = 2000
    learning_rate = 0.1
    eplise = 0.4

    #model,costJ = train_sgd( train_features, train_labels, iter_num, learning_rate, eplise )
    model,costJ,eplises = train_bgd( train_features, train_labels, iter_num, learning_rate, eplise )


    time_2 = time.time()
    print 'Traning cost: ',time_2 - time_1


    time_1 = time.time()

    test_pred = predction( test_features, model)

    score = accuracy_score( test_labels, np.array(test_pred) )

    print 'Test AUC: ', score

    time_2  = time.time()


    print 'predction cost: ',time_2 - time_1




