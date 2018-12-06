print("Loading libraries and functions")
import os
import sys
import json
os.chdir('source/use_feed_forward/')
sys.path.insert(0, '../../scripts/')
from functions import *
from feed_forward_nn import feed_forward_nn
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Nadam

print("Loading all data and preprocessing")

train = pd.read_csv('../../data/train.csv',dtype={'statement1':str, 'statement2':str})
features = pd.read_csv('../../data/features.csv')

if len(train)==len(features):
	rows_with_embeddings = np.where(~np.isnan(features.iloc[:,0]))[0]
	train = train.iloc[rows_with_embeddings,:]
	features = features.iloc[rows_with_embeddings]
else:
	raise ValueError('The number of rows in the features do not match the number of rows in the training set')

rows_with_valid_statements = np.where(~((train.statement1.isnull())|(train.statement2.isnull())))[0]
train = train.iloc[rows_with_valid_statements,:]
features = features.iloc[rows_with_valid_statements,:]



#stopwords = get_stopwords('../../data/stopwords.json')

print("Getting Embeddings")
embeddings = get_embeddings('../../data/use_embeddings.p')
s1_emb = np.zeros((len(train),512))
s2_emb = np.zeros((len(train),512))
for i in range(len(train)):
	s1_emb[i,:] = embeddings[train.statement1.iloc[i]]
	s2_emb[i,:] = embeddings[train.statement2.iloc[i]]
# train['statement1'] = train['statement1'].apply(str).apply(preprocess)
# train['statement2'] = train['statement2'].apply(str).apply(preprocess)

# print("Getting rare word features")
# statement1, statement2, rw_features = rare_word_handler(train, embeddings.keys(), stopwords)
# print("Computing Average embeddings for first column of statements")
# s1_emb, s1_found = get_all_average_embeddings(statement1, embeddings, stopwords)
# print("Computing Average embeddings for second column of statements")
# s2_emb, s2_found = get_all_average_embeddings(statement2, embeddings, stopwords)

# features = np.array(features)
# features = np.hstack([features, rw_features])

labels = np.reshape(np.array(train['resolution']),(len(train), 1))

study_objects, study_objects_all = get_study_object_lists(train)


# with open('hyperparameters.json', 'r') as fp:
#     config = json.load(fp)

# for j in range(len(config)):
kf=model_selection.KFold(5,shuffle=False,random_state=1)
loss = np.zeros((5,3))
i=0
print("Model: ",j,"/",len(config))
for itrain, iother in kf.split(study_objects):
	print("Model: ",i)

	x_train, labels_train, x_val, labels_val, x_test, labels_test = k_fold_split_helper_no_features(itrain, iother, study_objects, study_objects_all, s1_emb, s2_emb, labels)

	model = feed_forward_nn()
	opt = Adam()
	# if config[j]['optimizer'] == 'adam':
	# 	opt = Adam(lr = config[j]['learning_rate'])
	# else:
	# 	opt = Nadam(lr = config[j]['learning_rate'])
	model.compile(loss = "binary_crossentropy", optimizer = opt)
	#early_stopping = EarlyStopping(monitor = "val_loss", patience = 5)
	model_checkpoint = ModelCheckpoint('NN_test_model.h5',save_best_only=False,save_weights_only=True)
	hist = model.fit(x_train, labels_train,
					validation_data = (x_val, labels_val),
					epochs = 10, batch_size = 256, shuffle=True,
					callbacks=[model_checkpoint], verbose=2)

	min_val_loss_index = np.argmin(hist.history['val_loss'])
	train_loss = hist.history['loss'][min_val_loss_index]
	val_loss = hist.history['val_loss'][min_val_loss_index]
	
	model.load_weights('NN_test_model.h5')
	preds_train = model.predict(x_train)
	test_train = log_loss(labels_train, preds_train)
	preds_val = model.predict(x_val)
	val_loss = log_loss(labels_val, preds_val)
	preds_test = model.predict(x_test)
	test_loss = log_loss(labels_test, preds_test)
	print("Test loss = ", test_loss)
	
	loss[i,:] = [train_loss, val_loss, test_loss]
	i=i+1

	print("Average train, val, & test loss: ", np.mean(loss, axis=0))
	# config[j]['train_loss'] = np.mean(loss, axis=0)[0]
	# config[j]['val_loss'] = np.mean(loss, axis=0)[1]
	# config[j]['test_loss'] = np.mean(loss, axis=0)[2]
	# nn_dict = config[j]['nn_dict']
	# config[j].pop('nn_dict')
	# config[j].update(nn_dict)

	# current = pd.read_csv('test_results.csv',index_col=0)

	# df = pd.DataFrame(config[j], index=[0])
	# df = pd.concat([current,df],ignore_index=True)
	# df.to_csv('test_results.csv')

#fake_dat = np.reshape(np.concatenate([np.repeat(0,92),np.repeat(1,8)]),(100,1))
#fake_preds = np.reshape(np.repeat(.08,100),(100,1))
#log_loss(fake_dat,fake_preds)