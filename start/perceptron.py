# -*- coding:utf-8 -*-
#/usr/bin/python 
import numpy as np
import cv2
import random
import time
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score



def train_percp( x, y, iter_num, learning_rate ):
	#感知器核心算法部分
	#每次迭代随机抽取样本中一个样本对权重进行更新

	nrow, ncol = x.shape
	w = np.ones( (ncol,1) )
	b = 0

	correct = 0
	for i in xrange( iter_num ):

		index = random.randint( 0, nrow-1 )
		x_train = x[index]
		y_train = y[index]

		fx = np.dot( x_train, w )[0] * y_train + b
		if fx >= 0:
			correct += 1
			if correct > iter_num:
				break
			continue

		w = w + learning_rate*y_train*x_train
		b = b + learning_rate*y_train

	return ( w, b )


def predction( x, y, model):

	w, b = model
	lbls = [ ] 
	for i in range(len(x)):
		if np.dot( x[i], w )[0]+b > 0:
			lbls.append( 1 ) 
		else:
			lbls.append( -1 )

	return lbls


if __name__ == '__main__':

    print 'Start read data'

    time_1 = time.time()

    raw_data = pd.read_csv('/Users/fei_Daniel/Desktop/machineLearning-master/lihang_book_algorithm/data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]
    labels_new = []
    for line in labels:
    	if int(line) == 1:
    		labels_new.append( 1 )
    	else:
    		labels_new.append( -1 )
    labels_new = np.array( labels_new )


    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels_new, test_size=0.33, random_state=23323)


    iter_num = 5000
    learning_rate = 0.00001

    model = train_percp( train_features, train_labels, iter_num, learning_rate )
    lbls = predction( test_features, test_labels, model)


    score = accuracy_score( test_labels, np.array(lbls) )

    print score




