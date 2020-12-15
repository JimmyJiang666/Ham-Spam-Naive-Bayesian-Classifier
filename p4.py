import pandas as pd
import numpy as np
import math
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

tokenizer = RegexpTokenizer(r'\w+')

def tokens(str):
	return [t.lower() for t in set(tokenizer.tokenize(str))]

df = pd.read_csv('spam.csv', encoding = 'latin-1')

train, test = train_test_split(df, test_size=0.2)

# print(train.head())
# mytrain = train.loc[:,['v1','v2']]

dic = {0:[], 1:[]} #dictionary to store tokens with label : 0 is ham, 1 is spam

print(train)
count = 0
for item in train.values:
	list_of_token = tokens(item[1])
	label = item[0]
	if label == "ham":
		for token in list_of_token:
			dic[0].append(token)
	if label == "spam":
		for token in list_of_token:
			dic[1].append(token)
	# count += 1
	# if count > 10:
	# 	break

num_ham = len(dic[0])
num_spam = len(dic[1])
# print("num_ham: ", num_ham)
# print("num_spam: ", num_spam)

alpha = 0.1
#testing
#laplacing 
# for item in test.values:
"""
test = "Jimmy is a word that you never seen"
print(tokens(test))
log_ham_prob = 0
log_spam_prob = 0
for word in tokens(test):
	if dic[0].count(word) == 0 or dic[1].count(word) == 0:
		# This happens iff the word is unseen
		prob_ham = (alpha)/(num_ham)
		prob_spam = (alpha)/(num_spam)
		print("case 1: ")
		print("word is: ", word)
		print("prob_spam: ",prob_spam)
		print("prob_ham: ",prob_ham)
		log_ham_prob += np.log(prob_ham)
		log_spam_prob += np.log(prob_spam)
	else:
		prob_ham = dic[0].count(word)/(num_ham)
		prob_spam = dic[1].count(word)/(num_spam)
		print("case 2:")
		print("word is: ", word)
		print("prob_spam: ",prob_spam)
		print("prob_ham: ",prob_ham)
		log_ham_prob += np.log(prob_ham)
		log_spam_prob += np.log(prob_spam)
print("log_ham_prob: ", log_ham_prob)
print("log_spam_prob: ", log_spam_prob)
"""
rightOrWrong=[]
wrong_ham = 0
wrong_spam = 0
test_num_spam = 0
test_num_ham = 0
for item in test.values:
	msg = item[1]
	log_ham_prob = 0
	log_spam_prob = 0

	#we want to count the actual number of ham/spam msg
	if item[0] == "ham" :
		test_num_ham += 1
	else:
		test_num_spam += 1

	#bayes classifier 
	for word in tokens(msg):
		# prob_ham = (dic[0].count(word) + alpha)/(num_ham + k*alpha) #somehow this leads to a more accurate classification
		# prob_spam = (dic[1].count(word) + alpha)/(num_spam + k*alpha)
		prob_ham = (alpha)/(num_ham) if (dic[0].count(word) == 0) else dic[0].count(word)/(num_ham)  #treat unseen token differently with alpha
		prob_spam = (alpha)/(num_spam) if (dic[1].count(word)==0) else dic[1].count(word)/(num_spam)		
		log_ham_prob += np.log(prob_ham)
		log_spam_prob += np.log(prob_spam)

	if log_ham_prob > log_spam_prob: #if it is classified as ham
		if item[0] == "ham":
			rightOrWrong.append("correct")
		else:
			rightOrWrong.append("wrong")
			wrong_ham += 1
	elif log_ham_prob < log_spam_prob: #if it is classified as spam
		if item[0] == "ham":
			rightOrWrong.append("wrong")
			wrong_spam += 1
		else:
			rightOrWrong.append("correct")
	else:
		rightOrWrong.append("what?equal prob!")

# this is just for me to track correst or wrong table
# print(rightOrWrong)
# print(len(rightOrWrong))

print("number of actual ham in test:", test_num_ham)
print("number of actual spam in test:", test_num_spam)
print("fraction of ham identified as spam: ", wrong_spam/test_num_ham)
print("fraction of spam identified as ham: ", wrong_ham/test_num_spam)