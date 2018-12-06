import os
import sys
import pandas as pd
os.chdir('data')
sys.path.insert(0, '../scripts/')
from functions import *
from use_embeddings import get_use_embeddings
import pickle

train = pd.read_csv('train.csv')

statements = list(set(list(train.statement1.dropna()) + list(train.statement2.dropna())))
embeddings = get_use_embeddings(statements)

to_save = {}
for i in range(len(statements)):
	to_save[statements[i]] = embeddings[i]

with open('use_embeddings.p', 'wb') as fp:
	pickle.dump(to_save, fp)