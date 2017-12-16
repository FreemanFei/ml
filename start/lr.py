# -*- coding:utf-8 -*-
#/usr/bin/python

import pandas as pd
import numpy as np
import scipy as sp
#import matplotlib.pyplot as plt
import math
from sklearn.metrics import roc_curve, auc


def sigmoid( z ):
    return  1.0/(1+ np.exp(-z))

def train_sgd( x, y, iter_num, learning_rate, eplise ):
    # stochastic gradient descent
    nrow, ncol = x.shape
    # 在前面增加一个常数项, x0
    x = np.hstack( (np.array([1.0 for i in xrange(nrow)]).reshape(nrow,1),x ) )
    nrow, ncol = x.shape
    theta = np.zeros((ncol,1))

    costJ = []
    alpha = learning_rate
    lamda= 0.1
    k = 0
    i = 0
    for k in range( iter_num ):
        for i in range(nrow):
            x_sample = x[i]
            z = np.dot( x_sample, theta)
            h = 1.0/(1+ np.exp(-z))
            gradient = x_sample.reshape(ncol,1) * (y[i]-h) - alpha*lamda*theta / nrow
            theta += alpha*gradient
            i += 1
        J =  (np.sum((y-sigmoid(np.dot(x,theta)).reshape(nrow,1))**2))/(2*nrow) + lamda*np.sum( theta**2 )/(2*nrow)
        costJ.append( J )
        if np.sum( np.fabs(gradient) ) <= eplise:
            return theta, costJ
        k += 1
    return theta, costJ


def train_bgd( x, y, iter_num, learning_rate, eplise):
    # Batch Gradient Descent
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
    lamda = 0.1
    for k in range( iter_num ):
        z = np.dot( x, theta )
        h = sigmoid( z )
        J = ( np.sum(y - h)**2 )/( 2*nrow )  + lamda*np.sum( theta**2 )/(2*nrow)
        costJ.append( J )

        gradient = -np.dot( np.transpose(x), y-h ) / nrow - alpha*lamda*theta / nrow
        ep = sum( np.fabs(gradient) )
        eplises.append(ep)
        if ep < eplise:
            return theta, costJ, eplises

        #step = 0.001
        #a, b = get_ab_simple( x, y, theta, alpha, step, gradient, nrow )
        theta = theta + alpha * gradient
    return theta, costJ， eplises



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
    return theta, costJ， eplises




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


if __name__== '__main__':

    data_logistic = pd.read_csv('/data/home/gaopengfei/learn_scrip/machineLearning/LR/data_logistic.csv')
    data = np.array(data_logistic)
    nrow,ncol = data.shape
    x = data[:,0:(ncol-1)]
    y = data[:,ncol-1].reshape(nrow,1)
#    theta, costJ = train_sgd( x, y, 20, 0.005, 0.4)
    train_bgd( x, y, 100, 0.001,'')
