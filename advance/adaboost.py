#-*-coding:utf-8-*-
#/usr/bin/python
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score





def error_cal( x, y, w, col, cut_value):

	nrow, ncol = x.shape

	# y = 1 when the value is greater than cut_value
	results_great = np.ones(nrow) - 2
	# because y {-1, 1} so Gm {-1, 1} 
	results_great[ x[:,col] > cut_value ] = 1
	error_great_val = np.sum(np.fabs(y - results_great)*w )/2
	error_great_matrix  = np.fabs( y - results_great ) / 2


	# y = 1 when the value is less than cut_value
	results_less = np.ones(nrow) - 2
	results_less[ x[:,col] <= cut_value ] = 1
	error_less_val = np.sum(np.fabs((y - results_less))*w )/2
	error_less_matrix  = np.fabs( y - results_less ) / 2


	if error_great_val < error_less_val:
		error_val = error_great_val
		flag = 'great'
		error_matrix = error_great_matrix
		Gm = results_great
	else:
		error_val = error_less_val
		flag = 'less'
		error_matrix = error_less_matrix
		Gm = results_less

	return error_val, flag, error_matrix, Gm





def cut_stump( x, y, w):

	nrow, ncol = x.shape
	step = 20
	error_min = np.inf

	for col in xrange(ncol):
		col_value = list( set(x[:,col]) )
		max_value = max( col_value )
		min_value = min( col_value )
		step_siz = 1.0*( (max_value - min_value)/step )

		for i in xrange( -1, step+1 ):
			cut_value = min_value + 1.0*step_siz*i
			error_val, flag, error_matrix, Gm = error_cal( x, y, w, col, cut_value)

			if error_val < error_min:
				error_min = error_val
				best_col = col
				best_cut_value = cut_value
				best_flag = flag
				best_error_matrix = error_matrix
				best_Gm = Gm 

	return ( best_col, best_flag, best_cut_value ), error_min, best_error_matrix, best_Gm


def train_adaboost(x ,y, K):

	# Wm -> Gm
	# Gm -> em
	# em -> alpham
	# Gm & alpham -> Zm
	# Gm & alpham & Zm -> Wm+1


	nrow, ncol = x.shape
	#w为每条样本的贡献度 初始的每个样本权重为1/nrow
	w = np.ones( (1,nrow) ) / nrow

	all_trees = []
	all_alpha = []

	for k in xrange(K):
		#进行树桩分裂 在所有样本和特征下 只选取一个特种中的一个分裂点最为最优切分点

		tree, em, IGm, Gm = cut_stump(x, y, w)

		alpha = 0.5 * np.log((1-em)/em)

		all_trees.append( tree )
		all_alpha.append( alpha )
		# refresh Gm
		IGm[IGm==0] = -1
		z = w * np.exp( -alpha*y*Gm )
		w = z / np.sum(z)

	return all_trees, all_alpha


def get_predict(features_test , all_trees, all_alpha):
    predict = []
    num_trees = len(all_trees)
    for row in features_test:
        totol = 0
        for i in xrange(num_trees):
            col,flag,value = all_trees[i]
            if (flag == 'great' and row[col] > value) or (flag == 'less' and row[col] <= value):
                pre = 1
            else:
                pre = -1
            totol += all_alpha[i] * pre
        predict.append(np.sign(totol))
    return predict



if __name__ == '__main__':

	time_1 = time.time()
	raw_data = pd.read_csv('/Users/fei_Daniel/Desktop/my_ml/ml/data/horseColicTraining2.csv', header=0)
	raw_data_test = pd.read_csv('/Users/fei_Daniel/Desktop/my_ml/ml/data/horseColicTest2.csv', header=0)

	ncol = raw_data.shape[1]

	features = np.array( raw_data[raw_data.columns[0:ncol-1]] )
	label = np.array( raw_data['y'] )

	features_test = np.array( raw_data_test[raw_data_test.columns[0:ncol-1]] )
	label_test = np.array( raw_data_test['y'] )


	time_2 = time.time()
	print 'data processing cost: ',time_2 - time_1



	time_1 = time.time()

	all_trees, all_alpha = train_adaboost( features ,label , 1000)

	time_2 = time.time()
	print 'traning cost: ',time_2 - time_1


	time_1 = time.time()

	test_pre = get_predict( features_test , all_trees, all_alpha )

	time_2 = time.time()
	score = accuracy_score( label_test, np.array(test_pre) )
	print 'Test AUC: ',score

	print 'Predict cost: ',time_2 - time_1


