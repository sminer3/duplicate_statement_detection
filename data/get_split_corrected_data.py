import os
os.chdir('data/')
import pandas as pd
import numpy as np
import json

train = pd.read_csv('train.csv')
train['percent'] = train.respEqual / train.n
corrections = pd.read_csv('corrections.csv')
with open('statement_splits.json') as f:
	splits = json.load(f)

statements = list(set(list(train.statement1) + list(train.statement2)))
stmts_not_have = [s for s in statements if s not in splits.keys()]
for stmt in stmts_not_have:
	dict_to_add = {'sentences': [stmt], 'num_splits':1}
	splits[stmt] = dict_to_add

#Remove pairs where either of the statements has multiple ideas
stmt1_keep = np.array([True if splits[s]['num_splits']==1 else False for s in train.statement1])
print('statements not split from first statements:',np.sum(stmt1_keep))
stmt2_keep = np.array([True if splits[s]['num_splits']==1 else False for s in train.statement2])
print('statements not kept from second statements:',np.sum(stmt2_keep))
print('Total pairs not kept:',np.sum(stmt1_keep & stmt2_keep))

train_splits_removed = train[stmt1_keep & stmt2_keep].reset_index()
train_splits_removed.to_csv('train_splits_removed.csv',index=False, encoding='utf-8')

train_corrected_and_splits_removed = train_splits_removed
train_corrected_and_splits_removed[train_corrected_and_splits_removed['id_all'].isin(corrections.id_all)]
train = train[~train['id_all'].isin(corrections.id_all)]

#To apply corrections, set index to the id_all and then use the update function
