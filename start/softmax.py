#-*-coding:utf-8-*-
#!/usr/bin/python
import numpy as np
import pandas as pd
import time
import random
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_softmax( x, y, learing_rate, iter_num, lamda, lbls ):
    #softmax核心计算 每次迭代随机抽取一个样本进行梯度迭代的
    #采用L2正则

    nrow, ncol = x.shape
    k = len( set( y ) )
    w = np.zeros( (k,ncol+1) )

    for iteration in range( iter_num ):
        #随机抽取所有样本中的一个样本进行计算

        index = random.randint( 0, nrow-1)
        x_train = np.hstack( (x[ index ],[1] )) 
        y_train = y[ index ]

        first = 0
        gradient_list = []
        for lbl_id in range( k ):
            if y_train == lbls[ lbl_id ]:
                first = 1
            e = math.exp( np.dot( w[ lbl_id ], x_train ) )
            sum_e = sum( [math.exp( np.dot( w[ i ], x_train ) ) for i in range(k)] )

            gradient = 1.0*( -x_train*( first - 1.0*e/sum_e ) )/nrow+ lamda * w[ lbl_id ] 
            gradient_list.append( gradient )

            first = 0


        for i in range(k):
            w[ i ] = w[ i ] - learning_rate * gradient_list[ i ]

    return w




def prediction( x, y, w, lbls):

    test_predict = []
    for k in range(len(x)):
        feature = np.hstack( ( x[k],[1] ))

        tmp_dic = []
        for i in range( len(w) ):
            tmp_dic.append( np.dot( feature.T , w[i] ) )

        test_predict.append( lbls[ tmp_dic.index( max(tmp_dic) ) ] )

    return test_predict



if __name__ == '__main__':


    print 'Start'

    time_1 = time.time()

    raw_data = pd.read_csv('/Users/fei_Daniel/Desktop/machineLearning-master/lihang_book_algorithm/data/train.csv', header=0)
    data = raw_data.values

    imgs = np.array( data[0::, 1::] )
    labels = np.array( data[::, 0] )
    lbls = list( set( labels ) )


    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)

    time_2 = time.time()
    print 'data processing cost: ',time_2 - time_1


    time_1 = time.time()

    learning_rate = 0.01
    iter_num = 10000
    lamda = 0.01

    model = train_softmax( train_features, train_labels, learning_rate, iter_num, lamda, lbls )

    time_2 = time.time()
    print 'Traning cost: ',time_2 - time_1


    time_1 = time.time()

    test_predict = prediction( test_features, test_labels, model, lbls ) 
    score = accuracy_score(test_labels, test_predict)
    print 'Test AUC: ',score

    time_2 = time.time()
    print 'predction cost: ',time_2 - time_1



