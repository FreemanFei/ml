import time
import numpy as np
import random

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


def cost(x,y,w):
	all_cost = 0
	for k in range(len(y)):
		z = np.dot(w,x[k])
		h = 1.0 * ( 1.0/(1+ np.exp(-z)) )
		all_cost += ( y[k] - h )**2
	all_cost = all_cost/len(y)

	return float( all_cost )

def train_lr(x, y, iter_num, learning_rate):

	nrow, ncol = x.shape
	x = np.hstack( (np.array([1.0 for i in xrange(nrow)]).reshape(nrow,1),x) )
	nrow, ncol = x.shape
	w = np.zeros((1,ncol))
	costJ = []


	for i in xrange(iter_num):
		index = random.randint( 0,len(y)-1 )
		x_train = x[index]
		y_train = y[index]

		z = np.dot( w,x_train )
		h = 1.0 * ( 1.0/(1+ np.exp(-z)) )
		w = w + learning_rate*( y_train - h )*x_train / (2*nrow)

		J = cost(x,y,w)
		costJ.append(J)

	return w, costJ 


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



if __name__ == '__main__':

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

    iter_num = 100
    learning_rate = 0.1

    model,costJ = train_lr( train_features, train_labels, iter_num, learning_rate )


    time_2 = time.time()
    print 'Traning cost: ',time_2 - time_1


    time_1 = time.time()

    test_pred = predction( test_features, model)

    score = accuracy_score( test_labels, np.array(test_pred) )

    print 'Test AUC: ', score

    time_2	= time.time()


    print 'predction cost: ',time_2 - time_1





