# -*- coding:utf-8 -*-
#/usr/bin/python

import pandas as pd
import numpy as np
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score




def train_knn( x, y , k, test_features, test_labels, iter_num):
    pred_lbls = []
    test_lbls = []
    for break_num in range(iter_num):

        index = random.randint(0,len(test_features)-1)
        test_vec = test_features[index]

        test_lbls.append( test_labels[index] )


        knn_list = []

        for i in range(k):
            lbl = y[i]
            train_vec = x[i]

            dist = np.linalg.norm( train_vec - test_vec )
            knn_list.append( (dist,lbl) )


        knn_list = sorted( knn_list, reverse = True)
        for i in range(k,len(y)):
            lbl = y[i]
            train_vec = x[i]
            dist = np.linalg.norm( train_vec - test_vec )

            if knn_list[0][0] > dist:
                knn_list[0] = ( dist, lbl )

            knn_list = sorted( knn_list, reverse = True)

        all_lbl = []
        for dist, lbl in knn_list:
            all_lbl.append(lbl)

        single_lbl = list(set(all_lbl))
        max_value = 0
        max_lbl = -1

        print all_lbl
        print single_lbl


        for line in single_lbl:
            comp = all_lbl.count( line )
            print line, comp
            if comp >= max_value:
                max_value = comp
                max_lbl = line
        pred_lbls.append( max_lbl )


    return pred_lbls,test_lbls


if __name__ == '__main__':

    print 'Start'

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
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

    time_2 = time.time()
    print 'data processing cost: ',time_2 - time_1


    time_1 = time.time()

    pred_lbls,test_lbls = train_knn( train_features, train_labels, 10, test_features, test_labels, 100 )

    time_2 = time.time()
    print 'Traning cost: ',time_2 - time_1


    score = accuracy_score( test_lbls, pred_lbls )
    print "The accruacy socre is ", score



