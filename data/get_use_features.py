import os
import sys
import pandas as pd
os.chdir('data')
sys.path.insert(0, '../scripts/')
from functions import *
from word2vec_features import *
from simple_features import *
from fuzzy_features import *

embeddings = get_embeddings('use_embeddings.p')

train = pd.read_csv("train.csv")

s1_emb = np.zeros((len(train),512))
s2_emb = np.zeros((len(train),512))
for i in range(len(train)):
	if train.statement1[i] in embeddings.keys():
		s1_emb[i,:] = embeddings[train.statement1.iloc[i]]
	if train.statement2[i] in embeddings.keys():
		s2_emb[i,:] = embeddings[train.statement2.iloc[i]]

features = get_dist_features_all(s1_emb, s2_emb)

train['statement1'] = train['statement1'].apply(str).apply(preprocess)
train['statement2'] = train['statement2'].apply(str).apply(preprocess)

simple_features = get_simple_features_all(train['statement1'], train['statement2'])
features = pd.concat((features, simple_features),axis=1)

fuzzy_features = get_fuzzy_features_all(train['statement1'], train['statement2'])
features = pd.concat((features, fuzzy_features),axis=1)
features['id_all'] = train['id_all']
features.to_csv('use_features.csv')
