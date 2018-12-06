import pandas as pd
import numpy as np
from scipy.spatial import distance


def get_dist_features(s1_emb,s2_emb,return_names=True):
	#Function for calculating distance metrics between 2 word2vec cbow embeddings for 2 statements
	dist_features = np.zeros((1,6))
	dist_features[0,0] = distance.cosine(s1_emb, s2_emb)
	dist_features[0,1] = distance.cityblock(s1_emb, s2_emb)
	dist_features[0,2] = distance.jaccard(s1_emb, s2_emb)
	dist_features[0,3] = distance.euclidean(s1_emb, s2_emb)
	dist_features[0,4] = distance.canberra(s1_emb, s2_emb)
	dist_features[0,5] = distance.braycurtis(s1_emb, s2_emb)

	if return_names:
		column_names = ["cosine","manhattan","jaccard","euclidean","canberra","braycurtis"]
		return dist_features, column_names
	else:
		return dist_features

def get_dist_features_all(s1_emb_table, s2_emb_table):
	#Helper function for getting distance metrics for 2 tables of cbow embeddings
	if s1_emb_table.shape == s2_emb_table.shape:
		distances = []
		for i in range(len(s1_emb_table)):
			distances.append(get_dist_features(s1_emb_table[i], s2_emb_table[i], return_names=False))
		result = np.vstack(distances)
		result = pd.DataFrame(result)
		result.columns = ["cosine","manhattan","jaccard","euclidean","canberra","braycurtis"]
		return result
	else:
		raise ValueError("The dimensions of the 2 arrays are not the same")


