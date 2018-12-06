import os
os.chdir('data/Batches of pairs to correct')
import pandas as pd
import numpy as np
import json

train = pd.read_csv('../train.csv')
train['percent'] = train.respEqual / train.n
corrections = pd.read_csv('../corrections.csv')
train = train[~train['id_all'].isin(corrections.id_all)]
with open('../statement_splits.json') as f:
	splits = json.load(f)

statements = list(set(list(train.statement1) + list(train.statement2)))
stmts_not_have = [s for s in statements if s not in splits.keys()]
for stmt in stmts_not_have:
	dict_to_add = {'sentences': [stmt], 'num_splits':1}
	splits[stmt] = dict_to_add

#Remove pairs where either of the statements has multiple ideas
stmt1_keep = np.array([True if splits[s]['num_splits']==1 else False for s in train.statement1])
np.sum(stmt1_keep)
stmt2_keep = np.array([True if splits[s]['num_splits']==1 else False for s in train.statement2])
np.sum(stmt2_keep)
np.sum(stmt1_keep & stmt2_keep)

train_sub = train[stmt1_keep & stmt2_keep].reset_index()
low = .5
high = .85
train_sub = train_sub[(train_sub.percent >= low) & (train_sub.percent<=high)]
num_to_get = len(train_sub)
while num_to_get > 3700:
	low = low + .001
	train_sub = train_sub[(train_sub.percent >= low) & (train_sub.percent<=high)]
	num_to_get = len(train_sub)
train_sub = train_sub[(train_sub.percent >= low) & (train_sub.percent<=high)].sort_values('percent',ascending=False).reset_index()
keep = ['id_all', 'question', 'statement1', 'statement2','resolution','percent'] 
train_sub = train_sub.loc[:,keep].sort_values('id_all')
train_sub.to_csv('statement_pairs_to_correct2.csv',index=False)
