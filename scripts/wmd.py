#import os
#os.chdir('scripts')
from numpy import asarray, linalg
from pandas import read_csv
from functions import *

#embeddings = get_embeddings('../data/embeddings.p')
#stopwords = get_stopwords('../data/stopwords.json')

def wmd_approx(s1_list,s2_list,W):
	#Calculates the aprroximate word mover's distance using embeddings between two sentences by:
		#Calculating the minimum distance of each word from the first sentence to the second sentence and summing
		#Vice versa for the second sentence
		#Computes the average of the 2 sums
	#Inputs: The list of words from each sentence (cleaned and preprocessed),
		#The embeddings dictionary saved as W
	#All words from the list should be in W
	d1=0
	for i in s1_list:
		min_dist=100
		for j in s2_list:
			min_dist=min(min_dist,linalg.norm(W[i]-W[j]))
		d1 = d1 + min_dist
	d1 = d1 / len(s1_list) 
	d2=0
	for i in s2_list:
		min_dist=100
		for j in s1_list:
			min_dist=min(min_dist,linalg.norm(W[i]-W[j]))
		d2 = d2 + min_dist
	d2 = d2 / len(s2_list)
	return ((d1+d2)/2)


def calc_wmd(s1,s2,embeddings,stopwords):
	#Function that preprocesses text and returns wmd
	#Checks for empty sentences, preprocesses, removes stopwords, and removes words not in embeddings
	#If none of the words are in the embeddings, return None
	s1 = preprocess(s1)
	s2 = preprocess(s2)
	if s1=='' and s2=='':
		return None
	vect = (s1+' '+s2).split()
	word_emb_needed = list(set(vect)-set(stopwords))
	words_have = [w for w in word_emb_needed if w in embeddings.keys()]

	s1_list = [w for w in s1.split() if w in words_have]
	s2_list = [w for w in s2.split() if w in words_have]

	if len(s1_list)==0 or len(s2_list) == 0:
		return None
	else:
		return wmd_approx(s1_list,s2_list,embeddings)

def calc_wmd_table(table,embeddings,stopwords):
	#Function that calculates wmd for a table of pairs of statements
	#Preprocessing is taken care of with the calc_wmd function
	statement1 = table["statement1"].apply(str)
	statement2 = table["statement2"].apply(str)
	all_wmd=[0]*table.shape[0]
	for i in range(table.shape[0]):
		all_wmd[i]=calc_wmd(statement1.iloc[i],statement2.iloc[i],embeddings,stopwords)
		if i % 1000 == 0:
			print(i)
	return all_wmd

