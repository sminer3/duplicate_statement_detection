import pandas as pd
import numpy as np


def get_simple_features(s1,s2,return_names=True):
    #Get difference in character information
	simple_features = np.zeros((1,6))
	len1_chars = len(s1)
	len2_chars = len(s2)
	simple_features[0,0] = (len1_chars + len2_chars) / 2
	simple_features[0,1] = np.abs(len1_chars - len2_chars)

	#Get difference in word information
	len1_words = len(s1.split())
	len2_words = len(s2.split())
	simple_features[0,2] = (len1_words + len2_words) / 2
	simple_features[0,3] = np.abs(len1_words - len2_words)

	#Get common words
	simple_features[0,4] = len(set(s1.lower().split()).intersection(set(s2.lower().split())))
	simple_features[0,5] = simple_features[0,4] / ((len1_words + len2_words) / 2)

	if return_names:
		column_names = ['mean_characters','diff_characters','mean_n_words','diff_n_words','n_common_words','ratio_common_words']
		return simple_features, column_names
	else:
		return simple_features

def get_simple_features_all(s1_all, s2_all):
	if len(s1_all) == len(s2_all):
		all_simple_features = []
		for i in range(len(s1_all)):
			all_simple_features.append(get_simple_features(s1_all[i], s2_all[i], return_names=False))
		result = np.vstack(all_simple_features)
		result = pd.DataFrame(result)
		result.columns = ['mean_characters','diff_characters','mean_n_words','diff_n_words','n_common_words','ratio_common_words']
		return result
	else:
		raise ValueError("Number of statements from the 2 arguments are not equal")
