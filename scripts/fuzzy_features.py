import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

#Add fuzzy features
def get_fuzzy_features(s1,s2,return_names=True):
	fuzzy_features = np.zeros((1,7))
	
	fuzzy_features[0,0] = fuzz.QRatio(s1, s2)
	fuzzy_features[0,1] = fuzz.WRatio(s1, s2)
	fuzzy_features[0,2] = fuzz.partial_ratio(s1, s2)
	fuzzy_features[0,3] = fuzz.partial_token_set_ratio(s1, s2)
	fuzzy_features[0,4] = fuzz.partial_token_sort_ratio(s1, s2)
	fuzzy_features[0,5] = fuzz.token_set_ratio(s1, s2)
	fuzzy_features[0,6] = fuzz.token_sort_ratio(s1, s2)

	if return_names:
		column_names = ['QRatio','WRatio','partial_ratio','partial_token_set_ratio','partial_token_sort_ratio','token_set_ratio','token_sort_ratio']
		return fuzzy_features, column_names
	else:
		return fuzzy_features

def get_fuzzy_features_all(s1_all, s2_all):
	if len(s1_all) == len(s2_all):
		all_fuzzy_features = []
		for i in range(len(s1_all)):
			if i % 1000 == 0:
				print(i)
			all_fuzzy_features.append(get_fuzzy_features(s1_all[i], s2_all[i], return_names=False))
		result = np.vstack(all_fuzzy_features)
		result = pd.DataFrame(result)
		result.columns = ['QRatio','WRatio','partial_ratio','partial_token_set_ratio','partial_token_sort_ratio','token_set_ratio','token_sort_ratio']
		return result
	else:
		raise ValueError("Number of statements from the 2 arguments are not equal")
