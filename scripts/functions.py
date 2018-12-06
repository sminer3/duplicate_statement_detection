from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import json
import re
import string as string_lib
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import log_loss
from random import randint



#List of functions:
	#preprocess: preprocesses text
	#get_words: return a list of tokens from a list of statements
	
	#get_stopwords: reads in the stopwords from a filepath
	#get_embeddings: reads in the embeddings from a filepath
	
	#is_numeric: Checks if a string contains a number
	#prepare: prepares a statement by removing stopwords. Also removes words not in the embeddings and return those removed words in 2 lists
	#rare_word handler: uses 'prepare' to clean pairs of statements and output features on removed words

	#get_average_embedding: Get the average embedding for a statement; handles stop words and words not in the embedding
	#get_all_average_embeddings: Get the average embeddings for a list of statements and indicate which statements found no embeddings

	#get_study_object_lists: Function used during training to get data needed for splitting data by study objects
	#k_fold_split_helper: Function used during training to get splits from k fold into train, val, and test

def preprocess(input):
    #Remove punctuation (except for apostrophe's) and make lower case
	#Also removes non - printable characters (e.g. emoticons)
    string = str(input)
    string = string.lower()
    string = ''.join([w for w in string if w in string_lib.printable])
    replace = str.maketrans('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~', ' '*len('/\\"#$%&!()*+,-.:;<=>?@[]^_`{|}~'))
    string = string.translate(replace).strip()
    string = re.sub(' +',' ',string)
    #string = ' '.join([w for w in string.split() if w != 'a'])
    return str(string)


def get_words(sentences,with_preprocessing=True):
	#Input: list of sentences
	#Output: list of tokens
	sentences = set(sentences + list(map(preprocess, sentences)))
	vectorizer = CountVectorizer()
	vectorizer.fit(sentences)
	return list(vectorizer.vocabulary_.keys())


def get_stopwords(filepath):
	#Helper function that returns the stop words
	#Input: Filepath to the stopwords file
	with open(filepath) as f:
		stopwords = json.load(f)
	return stopwords['stopWords']

def get_embeddings(filepath):
	#Helper function that returns the embedding dictionary
	#Input: Filepath to the embeddings file
	with open(filepath, 'rb') as fp:
		embeddings = pickle.load(fp)
	return embeddings


def is_numeric(s):
	#Helper function that determines if any character in a string is numeric
    return any(i.isdigit() for i in s)

def prepare(s,words_have,stopwords):
    #Removes stopwords, words not in the embeddings (or otherwise specified by "words_have"), and words with numbers to get a clean statement
	#Returns the cleaned statement, along with 2 lists of words not found in the embeddings and words that contain numbers
    keep = []
    rare_words = []
    numbers = []
    for w in s.split()[::-1]:
        if w in words_have:
            keep = [w] + keep
        elif w not in stopwords:
            if is_numeric(w):
                numbers = [w] + numbers
            else:
                rare_words = [w] + rare_words
    clean_statement = " ".join(keep)
    return clean_statement, set(rare_words), set(numbers)


#prepare2
#Want to try to assign rare words a random vector that can be used in models
#Want number of rare word vectors to be a hyperparameter
def prepare2(s,words_have,stopwords,random_vec,rare_words_lookup,max_len):
	#Removes stopwords, words not in the embeddings (or otherwise specified by "words_have"), and words with numbers to get a clean statement
	#Returns the cleaned statement, along with 2 lists of words not found in the embeddings and words that contain numbers
	keep = []
	rare_words = []
	numbers = []
	for w in s.split()[::-1]:
		if w not in ['i','a','and','of','to']:
			keep = [w] + keep
		if w not in words_have and w not in stopwords:
			if w not in rare_words_lookup.keys():
				rare_words_lookup[w] = random_vec[randint(0,len(random_vec)-1),:]
			if is_numeric(w):
				numbers = [w] + numbers
			else:
				rare_words = [w] + rare_words
		if len(keep) == max_len:
			break
	clean_statement = " ".join(keep)
	return clean_statement, set(rare_words), set(numbers), rare_words_lookup


def rare_word_handler(df,words_have,stopwords, replace_rare_words=False, random_gen = 100, max_len = 15):
	#Input: table containing pairs, statement1 & statement2, words to include (usually from the embeddings), and the stop words
	#Output: statements with the stop words and uncommon words removed, with features on the intersection and union of rare words and words with numbers
	s1s = np.array([""] * len(df), dtype=object)
	s2s = np.array([""] * len(df), dtype=object)
	features = np.zeros((len(df), 4))
	words_have = list(words_have)
	random_vec = np.random.normal(size = (random_gen,300))
	rare_words_lookup = {}
	for i, (s1, s2) in tqdm(enumerate(list(zip(df["statement1"], df["statement2"])))):
		if i % 10000==0:
			print(i)
		if replace_rare_words:
			s1s[i], rare_words1, numbers1, rare_words_lookup = prepare2(s1,words_have,stopwords,random_vec,rare_words_lookup,max_len)
		else:
			s1s[i], rare_words1, numbers1 = prepare(s1,words_have,stopwords)
		if replace_rare_words:
			s2s[i], rare_words2, numbers2, rare_words_lookup = prepare2(s2,words_have,stopwords,random_vec,rare_words_lookup,max_len)
		else:
			s2s[i], rare_words2, numbers2 = prepare(s2,words_have,stopwords)
		features[i, 0] = len(rare_words1.intersection(rare_words2))
		features[i, 1] = len(rare_words1.union(rare_words2))
		features[i, 2] = len(numbers1.intersection(numbers2))
		features[i, 3] = len(numbers1.union(numbers2))
	if(replace_rare_words):
		return s1s, s2s, features, rare_words_lookup
	else:
		return s1s, s2s, features


def get_average_embedding(sentence,embeddings,stopwords):
	#Returns a 1xd vector containing an average of the available embeddings of all the words in a statement, stop words not included
	words=sentence.split()
	emb_length = len(embeddings[next(iter(embeddings))]) #Dimension of the embeddings
	emb = np.zeros((len(words), emb_length), dtype=object)
	remove_rows = []
	for i, word in enumerate(words):
		if word in embeddings.keys() and word not in stopwords:
			emb[i,:] = embeddings[word]	
		else:
			remove_rows.append(i)
	emb = np.delete(emb, remove_rows, 0)
	if(len(emb)>0):
		return np.mean(emb, 0)
	else:
		return np.zeros(emb_length)

def get_all_average_embeddings(statements,embeddings,stopwords):
    #Returns a nxd matrix containing an average of the embeddings of all the n words in the list of statements provided
    #Also returns a boolian vector stating whether none of the word embeddings were found for each of the statements 
    emb_length = len(embeddings[next(iter(embeddings))]) #Dimension of embeddings
    cbow_embeddings = np.empty((len(statements),emb_length))
    emb_found=np.ones(len(statements),dtype='bool')
    for i, statement in enumerate(statements):
        if i % 10000==0:
            print(i)
        cbow_embeddings[i] = get_average_embedding(statement,embeddings,stopwords)
        if np.mean(cbow_embeddings[i])==0:
            emb_found[i] = False
    return cbow_embeddings, emb_found

def embedding_helper(s1, s2, embeddings, maxlen=15):
	tokenizer = Tokenizer(filters='')
	tokenizer.fit_on_texts(embeddings.keys())
	word_index = tokenizer.word_index
	s1 = pad_sequences(tokenizer.texts_to_sequences(s1), maxlen=maxlen)
	s2 = pad_sequences(tokenizer.texts_to_sequences(s2), maxlen=maxlen)

	nb_words = len(word_index) + 1
	embedding_matrix = np.zeros((nb_words, 300))

	for word, i in word_index.items():
		embedding_vector = embeddings[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	return s1, s2, word_index, embedding_matrix



def get_study_object_lists(df):
	#Simple helper function to get a list of the unique study objects and a list of which study object each row belongs to
	#Uses questions in order to avoid problems with copied studies
	study_objects = df['question'].unique()
	study_objects_all = df['question']
	study_objects_all = pd.Series(list(study_objects_all)) #Work around solution to reset the index
	return study_objects, study_objects_all

def  k_fold_split_helper(itrain, iother, study_objects, study_objects_all, s1, s2, features, labels):
	#Helper function for splitting the data within the loop of k-fold cross validation:
	#Inputs: itrain, iother: splits from 
	ival, itest = model_selection.train_test_split(iother,test_size=.5,random_state=2)
	
	itrain=study_objects_all[study_objects_all.apply(lambda x: x in study_objects[itrain])].index
	ival=study_objects_all[study_objects_all.apply(lambda x: x in study_objects[ival])].index
	itest=study_objects_all[study_objects_all.apply(lambda x: x in study_objects[itest])].index

	s1_train = s1[itrain]
	s2_train = s2[itrain]
	f_train = features[itrain]
	labels_train = labels[itrain]
	print('Mean of training data: ', np.mean(labels_train))
	x_train = [s1_train, s2_train, f_train]

	s1_val = s1[ival]
	s2_val = s2[ival]
	f_val = features[ival]
	labels_val = labels[ival]
	print('Mean of validation data: ', np.mean(labels_val))
	x_val = [s1_val, s2_val, f_val]

	s1_test = s1[itest]
	s2_test = s2[itest]
	f_test = features[itest]
	labels_test = labels[itest]
	print('Mean of testing data: ', np.mean(labels_test))
	x_test = [s1_test, s2_test, f_test]

	return x_train, labels_train, x_val, labels_val, x_test, labels_test

def  k_fold_split_helper_no_features(itrain, iother, study_objects, study_objects_all, s1, s2, labels):
	#Helper function for splitting the data within the loop of k-fold cross validation:
	#Inputs: itrain, iother: splits from 
	ival, itest = model_selection.train_test_split(iother,test_size=.5,random_state=2)
	
	itrain=study_objects_all[study_objects_all.apply(lambda x: x in study_objects[itrain])].index
	ival=study_objects_all[study_objects_all.apply(lambda x: x in study_objects[ival])].index
	itest=study_objects_all[study_objects_all.apply(lambda x: x in study_objects[itest])].index

	s1_train = s1[itrain]
	s2_train = s2[itrain]
	labels_train = labels[itrain]
	print('Mean of training data: ', np.mean(labels_train))
	x_train = [s1_train, s2_train]

	s1_val = s1[ival]
	s2_val = s2[ival]
	labels_val = labels[ival]
	print('Mean of validation data: ', np.mean(labels_val))
	x_val = [s1_val, s2_val]

	s1_test = s1[itest]
	s2_test = s2[itest]
	labels_test = labels[itest]
	print('Mean of testing data: ', np.mean(labels_test))
	x_test = [s1_test, s2_test]

	return x_train, labels_train, x_val, labels_val, x_test, labels_test


def get_baseline_loss(avg_true):
	#Function that gives the baseline loss given the percent of 1/0 labels
	avg_true_round = np.round(avg_true,4)
	y_pred = np.repeat(avg_true_round,10000)
	y_actual = np.concatenate((np.repeat(1,avg_true_round*10000),np.repeat(0,10000-avg_true_round*10000)))
	return log_loss(y_actual, y_pred)
	
