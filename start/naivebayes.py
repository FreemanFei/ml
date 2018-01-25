#-*-coding:utf-8-*-
#!/usr/bin/python
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np



def re_sign( string ):
    return  re.sub('[^a-zA-Z0-9]',"",string).lower()

def load_txt( path_txt, path_dir):
    all_txt = []
    for line in path_txt:
        f = open(path_dir+line)
        tmp = []
        for line in f:
            for word in line.replace('\n','').split(' '):
                tmp.append( re_sign( word ) )
        tmp_new = []
        for word in tmp:
            if word != '':
                tmp_new.append(word)
        all_txt.append( ' '.join(tmp_new) )
    return all_txt

if __name__ == '__main__':

  ham_dir = '../data/email/ham/'
  spam_dir = '../data/email/spam/'
  test_dir = '../data/email/test/'
  path_ham = os.listdir(ham_dir)
  path_spam = os.listdir(spam_dir)
  path_test = os.listdir(test_dir)
  all_ham = load_txt( path_ham, ham_dir)
  all_spam = load_txt( path_spam, spam_dir)
  all_test = load_txt( path_test, test_dir)

  vectorizer=CountVectorizer()
  #vectorizer=CountVectorizer(stop_words='english')

  tmp_ham = ' '.join(all_ham)
  tmp_spam = ' '.join(all_spam)
  #tmp_test = ' '.join(all_test)
  tmp_all = [tmp_ham, tmp_spam]
  corpusTotoken=vectorizer.fit_transform(tmp_all).todense()
  corpusName = vectorizer.get_feature_names()


  #计算每一个出现的词的后验概率
  prob_spam = []
  prob_ham = []
  test = corpusTotoken.tolist()
  matrix_sum = []
  for i in range(len(test[0])):
      matrix_sum.append( int(test[0][i]) + int(test[1][i]) )
      
  for i in range( len(matrix_sum) ):
      try:
          prob_spam.append( test[1][i]/float(matrix_sum[i]) )
          prob_ham.append( test[0][i]/float(matrix_sum[i]) )
      except:
          prob_spam.append(0.0)
          prob_ham.append(0.0)

  #计算spam&ham的先验概率
  spam_prob = len(all_spam)/float( len(all_ham)+len(all_spam) )
  ham_prob = len(all_ham)/float( len(all_ham)+len(all_spam) )  


  #预测
  for i in range(len(all_test)):
    line = all_test[i]
    tmp_prob_spam = 0
    tmp_prob_ham = 0
    for word in line.split(' '):
        try:
            index = corpusName.index(word)
            tmp_prob_spam += (prob_spam[index]*spam_prob)/float( prob_spam[index]*spam_prob+ prob_ham[index]*ham_prob )   
            tmp_prob_ham += (prob_ham[index]*ham_prob)/float( prob_spam[index]*spam_prob+ prob_ham[index]*ham_prob )
        except:
            continue
    tmp_string = ''
    if tmp_prob_spam < tmp_prob_ham:
        tmp_string = 'ham'
    else:
        tmp_string = 'spam'
    print tmp_string, path_test[i]

           







