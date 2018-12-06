print("Loading libraries and functions")
import os
import sys
#os.chdir('source/logistic regression/')
sys.path.insert(0, '../../scripts/')
from functions import *
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np


train = pd.read_csv('../../data/train.csv')
features = pd.read_csv('../../data/features.csv')

if len(train)==len(features):
	rows_with_embeddings = np.where(~np.isnan(features.iloc[:,0]))[0]
	train = train.iloc[rows_with_embeddings,:]
	features = features.iloc[rows_with_embeddings]
else:
	raise ValueError('The number of rows in the features do not match the number of rows in the training set')

stopwords = get_stopwords('../../data/stopwords.json')
embeddings = get_embeddings('../../data/embeddings.p')

train['statement1'] = train['statement1'].apply(str).apply(preprocess)
train['statement2'] = train['statement2'].apply(str).apply(preprocess)

print("Getting rare word features")
statement1, statement2, rw_features = rare_word_handler(train, embeddings.keys(), stopwords)
print("Computing Average embeddings for first column of statements")
s1_emb, s1_found = get_all_average_embeddings(statement1, embeddings, stopwords)
print("Computing Average embeddings for second column of statements")
s2_emb, s2_found = get_all_average_embeddings(statement2, embeddings, stopwords)

features_save = features.copy().reset_index(drop=True)
rw_features_df = pd.DataFrame(rw_features,columns=['rw_intersect','rw_union','numbers_intersect','numbers_union'])
features_save = pd.concat([features_save, rw_features_df], axis=1)
features = np.array(features)
features = np.hstack([features, rw_features])

labels = np.reshape(np.array(train['resolution']),(len(train), 1))

study_objects, study_objects_all = get_study_object_lists(train)

#Split embeddings, features, and labels into train and test split by study objects
#Fit the absolute difference of the embeddings and the features
#Predict loss for test set
#Save test set with statements, questions, and features with labels and predictions

study_object_train, study_object_test = model_selection.train_test_split(study_objects,test_size=.1,random_state=0)
itrain=study_objects_all[study_objects_all.apply(lambda x: x in study_object_train)].index
itest=study_objects_all[study_objects_all.apply(lambda x: x in study_object_test)].index

features_train = np.hstack([np.abs(s1_emb[itrain] - s2_emb[itrain]),features[itrain]])
labels_train = labels[itrain]

features_test = np.hstack([np.abs(s1_emb[itest] - s2_emb[itest]), features[itest]])
labels_test = labels[itest]
statements_test = train.iloc[itest,:].loc[:,['question','statement1','statement2']].reset_index(drop=True)
features_save = features_save.iloc[itest,:].reset_index(drop=True)

print("Average matches in training set: ", np.mean(labels_train))
print("Average matches in test set: ", np.mean(labels_test))

model = LogisticRegression(C=.1)
model.fit(features_train,labels_train)

train_loss = log_loss(labels_train, model.predict_proba(features_train))
test_loss = log_loss(labels_test, model.predict_proba(features_test))
print('Training Loss: ', train_loss)
print('Test Loss: ', test_loss)

preds = np.reshape(model.predict_proba(features_test)[:,1], (len(features_test),1))

label_preds_df = pd.DataFrame(np.hstack([labels_test,preds]), columns = ['label', 'pred'])
results = pd.concat([statements_test, label_preds_df, features_save],axis=1)
results.to_csv('validation_results.csv',index=False)
