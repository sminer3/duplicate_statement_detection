import os
os.chdir('data/Batches of pairs to correct')
import pandas as pd
import numpy as np
import json

#Filename of corrected statements
filename = 'statement_pairs_to_correct_attempt1.csv'

#Combine and save
corrections_got = pd.read_csv(filename)
corrections_have = pd.read_csv('../corrections.csv')

corrections_got = corrections_got.drop('percent',axis=1)
corrections = pd.concat((corrections_got,corrections_have), axis=0)
corrections.to_csv('../corrections.csv',index=False,encoding='utf-8')



#Check how many pairs should have been removed but weren't from the first corrections that CST did

# with open('../statement_splits.json') as f:
# 	splits = json.load(f)

# statements = list(set(list(corrections_got.statement1) + list(corrections_got.statement2)))
# stmts_not_have = [s for s in statements if s not in splits.keys()]
# len(stmts_not_have)

# stmt1_keep = np.array([True if splits[s]['num_splits']==1 else False for s in corrections_got.statement1])
# np.sum(stmt1_keep)
# stmt2_keep = np.array([True if splits[s]['num_splits']==1 else False for s in corrections_got.statement2])
# np.sum(stmt2_keep)
# np.sum(stmt1_keep & stmt2_keep)
