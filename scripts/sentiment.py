import pandas as pd
import numpy as np
import indicoio
indicoio.config.api_key = '7607a2e29588d17b0626ed8b5c0206f5'

#Get sentiment of one statement
def get_sentiment(statement, lookup_table, save_results = False):
	if save_results:
		if statement in lookup_table.index:
			return lookup_table.loc[statement,'sentiment_hq'], False
		else:
			print('Result not found in table')
			return indicoio.sentiment_hq(statement), True
	else:
		if statement in lookup_table.index:
			return lookup_table.loc[statement,'sentiment_hq']
		else:
			print('Result not found in table')
			return indicoio.sentiment_hq(statement)
	
#Get sentiment of 2 lists of statements
def get_sentiment_all(s1_all, s2_all, lookup_table, update_table = False):
	if len(s1_all) == len(s2_all):
		if update_table:
			all_sentiments = []
			num_added = 0
			for i in range(len(s1_all)):
				if i % 1000 == 0:
					print(i)
				s1_sentiment, s1_not_found = get_sentiment(s1_all[i], lookup_table, save_results = True)
				s2_sentiment, s2_not_found = get_sentiment(s2_all[i], lookup_table, save_results = True)
				if s1_not_found:
					to_add = pd.DataFrame(s1_sentiment, columns = ['sentiment_hq'], index = [s1_all[i]])
					lookup_table = lookup_table.append(to_add)
					num_added = num_added + 1
				if s2_not_found:
					to_add = pd.DataFrame(s2_sentiment, columns = ['sentiment_hq'], index = [s2_all[i]])
					lookup_table = lookup_table.append(to_add)
					num_added = num_added + 1
				all_sentiments.append(np.array([[s1_sentiment, s2_sentiment]]))
			result = np.vstack(all_sentiments)
			result = pd.DataFrame(result)
			result.columns = ['sentiment1', 'sentiment2']
			print(num_added, 'statements added to the lookup table')
			return result, lookup_table
		else:
			all_sentiments = []
			for i in range(len(s1_all)):
				if i % 1000 == 0:
					print(i)
				s1_sentiment = get_sentiment(s1_all[i], lookup_table)
				s2_sentiment = get_sentiment(s2_all[i], lookup_table)
				all_sentiments.append(np.array([[s1_sentiment, s2_sentiment]]))
			result = np.vstack(all_sentiments)
			result = pd.DataFrame(result)
			result.columns = ['sentiment1','sentiment2']
			return result
	else:
		raise ValueError("Number of statements from the 2 arguments are not equal")
