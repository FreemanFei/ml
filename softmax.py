#-*-coding:utf-8-*-
#!/usr/bin/python
import numpy as np
from sklearn import preprocessing
import pandas as pd
import time
import random
import math

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
    correct = 0
    for k in range(len(x)):
        feature = np.hstack( ( x[k],[1] ))

        tmp_dic = []
        for i in range( len(w) ):
            tmp_dic.append( np.dot( feature.T , w[i] ) )

        if lbls[ tmp_dic.index( max(tmp_dic) ) ] == y[k]:
            correct += 1
   
    print 'Test AUC: ', 1.0* correct/len(y)

if __name__ == '__main__':


    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('/data/home/gaopengfei/learn_scrip/lihang_book_algorithm/data/train.csv', header=0)
    data = raw_data.values

    imgs = np.array( data[0::, 1::] )
    labels = np.array( data[::, 0] )
    lbls = list( set( labels ) )

    learning_rate = 0.01
    iter_num = 1000000
    lamda = 0.01

   
    model = train_softmax( imgs, labels, learning_rate, iter_num, lamda, lbls )

    prediction( imgs, labels, model, lbls ) 


