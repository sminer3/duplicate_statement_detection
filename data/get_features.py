import os
import sys
import pandas as pd
os.chdir(os.getcwd() + '/data')
sys.path.insert(0, '../scripts/')
from functions import *
from wmd import *
from word2vec_features import *
from simple_features import *
from fuzzy_features import *

stopwords = get_stopwords('stopwords.json')
embeddings = get_embeddings('embeddings.p')

train = pd.read_csv("train.csv")
wmd = calc_wmd_table(train)
wmd = pd.DataFrame(wmd,columns = ['wmd'])

train['statement1'] = train['statement1'].apply(str).apply(preprocess)
train['statement2'] = train['statement2'].apply(str).apply(preprocess)

s1_emb, s1_found = get_all_average_embeddings(train['statement1'], embeddings, stopwords)
s2_emb, s2_found = get_all_average_embeddings(train['statement2'], embeddings, stopwords)

dist_features = get_dist_features_all(s1_emb, s2_emb)
features = pd.concat((wmd,dist_features),axis=1)

simple_features = get_simple_features_all(train['statement1'], train['statement2'])
features = pd.concat((features, simple_features),axis=1)

fuzzy_features = get_fuzzy_features_all(train['statement1'], train['statement2'])
features = pd.concat((features, fuzzy_features),axis=1)
features['id_all'] = train['id_all']
features.to_csv('features.csv')

