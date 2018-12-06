from functions import *
from wmd import *
from word2vec_features import *
from simple_features import *
from fuzzy_features import *
from sentiment import *
import pandas as pd
import numpy as np

#stopwords = get_stopwords('../data/stopwords.json')
#embeddings = get_embeddings('../data/embeddings.p')
#sentiment_look_up = pd.read_csv('../data/indico_sentiment_hq_key.csv',index_col=1)

def calc_features(s1,s2,sentiment_look_up,embeddings,stopwords, return_embeddings = False):
	sent1 = get_sentiment(s1, sentiment_look_up)
	sent2 = get_sentiment(s2, sentiment_look_up)
	sentiment_features = np.array([[np.abs(sent1-sent2), np.mean((sent1,sent2))]])
	
	wmd = np.reshape(calc_wmd(s1,s2,embeddings,stopwords),(1,1))
	s1 = preprocess(s1)
	s2 = preprocess(s2)

	s1_emb = get_average_embedding(s1, embeddings, stopwords)
	s2_emb = get_average_embedding(s2, embeddings, stopwords)

	dist_features = get_dist_features(s1_emb, s2_emb, return_names = False)
	simple_features = get_simple_features(s1, s2, return_names = False)
	fuzzy_features = get_fuzzy_features(s1, s2, return_names = False)
	
	rw_features = np.zeros((1, 4))
	s1_clean, rare_words1, numbers1 = prepare(s1,embeddings.keys(), stopwords)
	s2_clean, rare_words2, numbers2 = prepare(s2,embeddings.keys(), stopwords)
	rw_features[0, 0] = len(rare_words1.intersection(rare_words2))
	rw_features[0, 1] = len(rare_words1.union(rare_words2))
	rw_features[0, 2] = len(numbers1.intersection(numbers2))
	rw_features[0, 3] = len(numbers1.union(numbers2))

	features = np.concatenate((wmd, dist_features, simple_features, fuzzy_features, rw_features, sentiment_features), axis=1)
	if return_embeddings:
		return np.reshape(s1_emb,(1,300)), np.reshape(s2_emb,(1,300)), features
	else:
		return features 


