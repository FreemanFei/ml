#-*-coding:utf-8-*-
#/usr/bin/python

import numpy as np




def regressErr(dataset):
    '''
    输入：数据集(numpy.mat类型)
    功能：求数据集划分左右子数据集的误差平方和之和
    输出: 数据集划分后的误差平方和
    '''
    #由于回归树中用输出的均值作为叶节点，所以在这里求误差平方和实质上就是方差
    return np.var(dataset[:,-1]) * np.shape(dataset)[0]

def regressLeaf(dataset):
    '''
    输入：数据集
    功能：求数据集输出列的均值
    输出：对应数据集的叶节点
    '''
    return np.mean(dataset[:,-1])


def split_dataset(data_array, col, val):
	array_1 = data_array[data_array[:,col] >= val,:]
	array_2 = data_array[data_array[:,col] < val,:]
	return array_1, array_2


def CartTree(data_array):

	thresholdErr = 1
	#如果切分的数据小于这个值 则不去计算
	thresholdSamples = 4

	#当数据中输出值都相等时，feature = None,value = 输出值的均值（叶节点）
	if len(set(data_array[:,-1])) == 1:
		return None,regressLeaf(data_array)

	nrow, ncol = data_array.shape
	Err = regressErr(data_array)

	bestErr = np.inf; bestCol = 0; bestValue = 0

	for col in xrange( ncol - 1 ):
		value = list(set( data_array[:,col] ))
		for val in value:
			left, right = split_dataset(data_array , col, val)
			if (left.shape[0] < thresholdSamples) or (right.shape[0] < thresholdSamples): continue

			tmpErr = regressErr(left) + regressErr(right)

			if tmpErr < bestErr:
				bestErr = tmpErr
				bestValue = val
				bestCol = col

	#检验在所选出的最优划分特征及其取值下，误差平方和与未划分时的差是否小于阈值，若是，则不适合划分
	if (Err - bestErr) < thresholdErr:
		return None,regressLeaf(data_array)
	return bestCol,bestValue


def build_tree(data_array):
	bestCol,bestValue = CartTree(data_array)
	if bestCol == None:
		return node(results = bestValue)
	else:
		array_1,array_2 = split_dataset(data_array,bestCol,bestValue)
		greater_branch = build_tree(array_1)
		less_branch = build_tree(array_2)
		return node(col = bestCol,value = bestValue,gb = greater_branch ,lb = less_branch )



def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat

class node:
    def __init__(self,col = -1,value = None, results = None, gb = None, lb = None):
        self.col = col
        self.value = value
        self.results = results
        self.gb = gb
        self.lb = lb


def printtree(tree,indent=''):
   # Is this a leaf node?
   if tree.results!=None:
      print str(tree.results)
   else:
      # Print the criteria
      print str(tree.col)+':'+str(tree.value)

      # Print the branches
      print indent+'T->',
      printtree(tree.gb,indent+'  ')
      print indent+'F->',
      printtree(tree.lb,indent+'  ')


if __name__ == '__main__':

	data = loadDataSet('/Users/fei_Daniel/Desktop/machineLearning-master/CART/ex0.txt')
	data_array = np.array(data)

	tree = build_tree(data_array)

	print printtree(tree)



