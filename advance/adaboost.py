#-*-coding:utf-8-*-
#/usr/bin/python
import pandas as pd
import numpy as np
import time



def error_cal( x, y, w, col, cut_value):

	nrow, ncol = x.shape

	# y = 1 when the value is greater than cut_value
	results = np.zeros(nrow) - 2
	results[ x[:,col] > cut_value ] = 1
	error_great_val = np.sum( (y - results)*w ) / 2
	error_great_matrix  = np.fabs( y - results ) / 2

	# y = 1 when the value is less than cut_value
	results = np.zeros(nrow) - 2
	results[ x[:,col] <= cut_value ] = 1
	error_less_val = np.sum( (y - results)*w ) / 2
	error_less_matrix  = np.fabs( y - results ) / 2


	if error_great_val < error_less_val:
		error_val = error_great_val
		flag = 'great'
		error_matrix = error_great_matrix
	else:
		error_val = error_less_val
		flag = 'less'
		error_matrix = error_less_matrix

	return error_val, flag, error_matrix





def cut_stump( x, y, w):

	nrow, ncol = x.shape
	step = 10
	error_min = np.inf

	for col in xrange(ncol):
		col_value = list( set(x[:,col]) )
		max_value = max( col_value )
		min_value = min( col_value )
		step_siz = 1.0*( (max_value - min_value)/step )

		for i in xrange( -1, step+1 ):
			cut_value = min_value + 1.0*step_siz*i
			error_val, flag, error_matrix = error_cal( x, y, w, col, cut_value)

			if error_val < error_min:
				error_min = error_val
				best_col = col
				best_cut_value = cut_value
				best_flag = flag
				best_error_matrix = error_matrix

	return ( best_col, best_flag, best_cut_value ), error_min, best_error_matrix


def train_adaboost(x ,y, K):

	nrow, ncol = x.shape
	#w为每条样本的贡献度
	w = np.ones( (nrow,1) ) / nrow

	all_trees = []
	all_a = []

	for k in xrange(K):
		#进行树桩分裂 在所有样本和特征下 只选取一个特种中的一个分裂点最为最优切分点

		tree, best_error_value, best_error_matrix = cut_stump(x, y, w)

		a = 0.5 * np.log((1-best_error_value)/best_error_value)

		all_trees.append( tree )
		all_a.append( a )

		best_error_matrix[best_error_matrix==0] = -1
		break



if __name__ == '__main__':

	time_1 = time.time()
	raw_data = pd.read_csv('/Users/fei_Daniel/Desktop/my_ml/ml/data/horseColicTraining2.csv', header=0)
	#print raw_data

	ncol = raw_data.shape[1]
	print ncol

	features = np.array( raw_data[raw_data.columns[0:ncol-1]] )
	label = np.array( raw_data['y'] )


	time_2 = time.time()
	print 'data processing cost: ',time_2 - time_1



	time_1 = time.time()

	train_adaboost( features ,label , 500)

	time_2 = time.time()
	print 'traning: ',time_2 - time_1


