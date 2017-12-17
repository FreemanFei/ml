# -*- coding:utf-8 -*-
#/usr/bin/python 
import numpy as np
import random
import time
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score






def train_nb( x, y):

	#naives bayes的核心算法 




if __name__ == '__main__':

	print 'Start'

	time_1 = time.time()

    raw_data = pd.read_csv('/Users/fei_Daniel/Desktop/machineLearning-master/lihang_book_algorithm/data/train.csv', header=0)
    data = raw_data.values

    imgs = data[0::, 1::]
    labels = data[::, 0]

    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels_new, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print 'data processing cost: ',time_2 - time_1




