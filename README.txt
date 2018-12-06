Folders:

#Raw (raw data)
	-final_matches.csv is matches collected from early groupsolver studies (SQL)
	-documentdb_matches.csv is matches collected from studies done more recently (Azure documentdb)

#Data
	1. In order to get data ready for training run get_clean_data.R. Output file = train.csv
	2. To get word embeddings for words in the training data, run get_word2vec.py. Output file = embeddings.p
	3. To get features that are used in training run get_features.py (works in runtime) Output file = features.csv
		-Uses functions from scripts folder
	4. To get indico sentiment features:
		a. Run get_indico_sentiment_hq.py to get sentiment lookup table for all statements in training data Output file: indico_sentiment_hq_key.csv (function currently broken)
		b. Run get_sentiment_features.py to get sentiment features for every row in the training data Output file: sentiment_features.py
	5. stopwords.json contains a list of stopwords (words that we want to remove before training because the statement does not add to the sentence meaning)


	A couple of other considerations include using corrected data (some matches are incorrectly labeled. CST has fixed a few statements, but not all),
	data that has split statements removed, and the universal sentence encoder.
	
	5. For corrections:
		a. Batches of pairs to correct folder includes scripts for producing data for CST that has not already been corrected
		b. corrections.csv are the current corrections we have
		c. Run data_corrector.R to get training data with corrections. Output file = corrected_train.csv
	6. For split statements removed:
		a. Run statement_splitter.py in order to get split data for the training data. Output file = statement_splits.json
		b. Run get_split_corrected_data.py to get data that was not split Output file = train_splits_removed.csv
	7. For the universal sentence encoder (better than simple average sentence embeddings):
		a. Run get_use_embedding_features.py to get a dictionary of the embeddings for each sentence in the training set. Output file = use_embeddings.p
		b. Run get_use_embedding_features.py to get the features using use instead of word2vec embeddings. Output file = use_features.csv

#Analysis:
	Python notebooks. Does a basic analysis on the raw documentdb data and another analysis on the training sets with splits removed vs. not removed.

#Source
	4 general model frameworks considered. 
	Used logistic regression to do a simple evaluation of the features
	Feed forward network is what we use now. (There's actually a lot of room for improvement using the other models, especially when focusing on log loss)
	use_feed_forward is a feed forward network using the universal sentence encoder. Outperforms the regular feed forward (which uses word2vec)
	Decomposable Attention model (from a paper by google https://arxiv.org/abs/1606.01933) This one uses word2vec with a more complicated network. Also outperforms regular feed forward network

	#Logistic regression (with regularization to improve performance)
		4 tests
		1. train.py runs vanilla logistic regression with regularization
		2. train_use.py uses the universal sentence encoder (improved performance)
		3. train_non_split_data.py trains using the training set with split statements removed (improvement)
		4. train_test_sentiment.py Tests using sentiment as well as other features (marginal improvement)

	#Feed forward network
		Since there are many hyperparameters in neural networks, folder is designed to test different hyperparameters
		1. generate_hpyerparameters.py genderates a random set of hyperparameters to use for testing
		2. feed_forward_nn.py is the neural network code
		3. train.py tests the neural network using the hyperparameters from hyperparameters.json.
		4. test_results.csv shows how well each set of hyperparameters did

	#use_feed_forward
		Same as feed forward but uses universal sentence encoded embeddings
		Didn't go through the work to look at hyperparameters (used a set of default hyperparameters)

	#Decomposable Attention
		decomposable_attention.py is the code for the decomposable_attention network
		Also didn't go through the work for hyperparameters
	
#Scripts
	Helper functions commonly used in other scripts
	1. calc_features.py (didn't use as much)
	2. docdb_query: used to get data from document_db
	3. function.py: Several functions uses for preprocessing and training models, decriptions of function in comments in file
	4. fuzzy_features.py: types of text features called "fuzzy" features that seem to improve model performance
	5. sentiment.py: Used to help get sentiment features
	6. simple_features.py: Used to calculate simple text features (e.g. text length differences, common word count, etc.)
	7. use_embeddings.py Loads univeral sentence encoder model
	8. wmd.py: calculates word mover's distance
	9. word2vec_features.py: Used to calculate word embedding features (can be used for universal sentence encoded sentences as well)

