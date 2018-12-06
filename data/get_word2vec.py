#Load Libraries
import gensim
import pandas as pd
import sys
import os
import pickle
#os.chdir(os.getcwd() + '/data')
sys.path.insert(0, '../scripts/')
from functions import *

#Load Google Word2Vec
print("Loading google word2vec")
path_to_google_word2vec = 'C:/Users/sminer/Downloads/GoogleNews-vectors-negative300.bin.gz' #Change as needed
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_google_word2vec, binary=True)

#Load  training data and extract words
dat = pd.read_csv('train.csv')
sentences = list(set(list(dat.statement1.values.astype('U')) + list(dat.statement2.values.astype('U'))))
print("Extracting unique words")
words = get_words(sentences)

#Get embeddings
embeddings = dict()
for w in words:
	if w in model.vocab:
		embeddings[w] = model[w]

with open('embeddings.p', 'wb') as fp:
	pickle.dump(embeddings, fp)

print("Embeddings saved")
