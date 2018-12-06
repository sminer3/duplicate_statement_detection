import os
import sys
import pandas as pd
os.chdir(os.getcwd() + '/data')
sys.path.insert(0, '../scripts/')
from functions import *
from sentiment import *

train = pd.read_csv('train.csv')
lookup_table = pd.read_csv('indico_sentiment_hq_key.csv',index_col=1)

sentiment_features, lookup_table = get_sentiment_all(train.statement1,train.statement2, lookup_table, update_table = True)
lookup_table.to_csv('indico_sentiment_hq_key.csv')
sentiment_features.to_csv('sentiment_features.csv',index=False)