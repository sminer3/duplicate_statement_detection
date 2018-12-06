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
features = pd.read_csv('../../data/features.csv',index_col=0)
features = features.drop('id_all',axis=1)


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

features = np.array(features)
features = np.hstack([features, rw_features])

labels = np.reshape(np.array(train['resolution']),(len(train), 1))

study_objects, study_objects_all = get_study_object_lists(train)
kf=model_selection.KFold(5,shuffle=False,random_state=1)
#c_values = [.1,2,32]
all_losses = np.zeros((len(c_values), 3))
all_auc = np.zeros((len(c_values),3))
for j, c in enumerate(c_values):
	i=0
	loss = np.zeros((5,3))
	auc = np.zeros((5,3))
	for itrain, iother in kf.split(study_objects):
		print("Model: ",i)

		x_train, labels_train, x_val, labels_val, x_test, labels_test = k_fold_split_helper(itrain, iother, study_objects, study_objects_all, s1_emb, s2_emb, features, labels)
		x_train = np.hstack([np.abs(x_train[0] - x_train[1]), x_train[2]])
		x_val = np.hstack([np.abs(x_val[0] - x_val[1]), x_val[2]])
		x_test = np.hstack([np.abs(x_test[0] - x_test[1]), x_test[2]])

		model = LogisticRegression(C=c)
		model.fit(x_train,labels_train)

		train_loss = log_loss(labels_train, model.predict_proba(x_train))
		val_loss = log_loss(labels_val, model.predict_proba(x_val))
		test_loss = log_loss(labels_test, model.predict_proba(x_test))

		train_auc = roc_auc_score(labels_train, model.predict_proba(x_train)[:,1])
		val_auc = roc_auc_score(labels_val, model.predict_proba(x_val)[:,1])
		test_auc = roc_auc_score(labels_test, model.predict_proba(x_test)[:,1])

		loss[i,:] = [train_loss, val_loss, test_loss]
		auc[i,:] = [train_auc, val_auc, test_auc]
		i=i+1

	print("Average train, val, & test loss: ", np.mean(loss, axis=0))
	print("Average train, val, & test auc: ", np.mean(auc, axis=0))
	all_losses[j,:] = np.mean(loss, axis=0)
	all_auc[j,:] = np.mean(auc, axis=0)

results = pd.DataFrame(all_losses, columns = ["Training Loss", "Validation Loss", "Test Loss"], index = c_values)
results.to_csv('lr_losses_with_l2_regularization4.csv')


